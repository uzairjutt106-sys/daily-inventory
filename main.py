import sqlite3
from datetime import date
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends, Query, Path as FPath
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Security Configuration ---
API_KEY = "@uzair143"

async def get_api_key(x_api_key: str = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=400, detail="X-API-Key header required")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

DATABASE_FILE = "daily_inventory.db"

def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT NOT NULL,
                purchase_rate REAL,
                sale_rate REAL,
                quantity_kg REAL NOT NULL,
                transaction_date TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(transaction_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_item ON transactions(item_name);")
        conn.commit()
        print(f"Database initialized and table 'transactions' is ready in {DATABASE_FILE}")
    except sqlite3.Error as e:
        print(f"Error during database initialization: {e}")
    finally:
        if conn:
            conn.close()

init_db()

# ---------- Models ----------
class Transaction(BaseModel):
    item_name: str = Field(..., example="copper")
    purchase_rate: Optional[float] = Field(None, ge=0, example=2200, description="Cost price per kg. Can be null.")
    sale_rate: Optional[float] = Field(None, ge=0, example=25000, description="Selling price per kg. Can be null.")
    quantity_kg: float = Field(..., gt=0, example=5.0, description="Quantity of the item in kilograms.")
    transaction_date: Optional[date] = Field(None, description="If omitted, defaults to today's date")

class DeleteResponse(BaseModel):
    deleted_id: int
    message: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title="Daily Inventory API",
    description="API for recording daily purchase and sale transactions using SQLite.",
    version="1.1.0",
    lifespan=lifespan
)

# ✅ --- Enable CORS for Next.js frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://192.168.0.108:4001",
        "http://192.168.0.108:3000",
        "http://192.168.0.108:4000",
        "http://localhost:4000",
        "http://127.0.0.1:4000",
        "http://192.168.0.113:4000",
        "http://localhost:4001",
        "http://127.0.0.1:4001",
        "http://192.168.0.113:4001",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.0.113:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- UI serving ----------
base_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=base_dir), name="static")

@app.get("/", include_in_schema=False)
async def serve_ui():
    html_path = base_dir / "inventory.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(
        f"""<!doctype html><meta charset='utf-8'>
        <h1>inventory.html not found</h1>
        <p>Expected at: <code>{html_path}</code></p>
        <p>Put your <strong>inventory.html</strong> next to this Python file or open
           <a href="/static/inventory.html">/static/inventory.html</a>.</p>""",
        status_code=200
    )

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    ico = base_dir / "favicon.ico"
    if ico.exists():
        return FileResponse(ico)
    return HTMLResponse("", status_code=204)

@app.get("/health", include_in_schema=False)
async def health():
    return {"ok": True}

# --------- API Endpoints ----------

@app.post("/transactions", status_code=201)
async def record_transaction(transaction: Transaction, api_key: str = Depends(get_api_key)):
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        tx_date = (transaction.transaction_date or date.today()).isoformat()

        data = (
            transaction.item_name,
            transaction.purchase_rate,
            transaction.sale_rate,
            transaction.quantity_kg,
            tx_date
        )

        cursor.execute("""
            INSERT INTO transactions (item_name, purchase_rate, sale_rate, quantity_kg, transaction_date)
            VALUES (?, ?, ?, ?, ?)
        """, data)
        conn.commit()

        # ✅ Calculate profit only if both rates exist
        if transaction.sale_rate is not None and transaction.purchase_rate is not None:
            unit_profit = transaction.sale_rate - transaction.purchase_rate
            total_profit = unit_profit * transaction.quantity_kg
        else:
            total_profit = None

        return {
            "message": "Transaction recorded successfully",
            "item_name": transaction.item_name,
            "quantity_kg": transaction.quantity_kg,
            "date": tx_date,
            "total_profit": round(total_profit, 2) if total_profit is not None else None
        }
    except sqlite3.Error as e:
        print(f"Database error during insertion: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not record transaction.")
    finally:
        if conn:
            conn.close()

@app.get("/transactions")
async def get_all_transactions(
    item_name: Optional[str] = Query(None),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        where, params = [], []
        if item_name:
            where.append("item_name = ?")
            params.append(item_name)
        if date_from:
            where.append("transaction_date >= ?")
            params.append(date_from.isoformat())
        if date_to:
            where.append("transaction_date <= ?")
            params.append(date_to.isoformat())
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        cursor.execute(f"SELECT COUNT(*) AS c FROM transactions {where_sql}", params)
        total = cursor.fetchone()["c"]
        cursor.execute(f"""
            SELECT id, item_name, purchase_rate, sale_rate, quantity_kg, transaction_date
            FROM transactions
            {where_sql}
            ORDER BY transaction_date DESC, id DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset])
        rows = [dict(row) for row in cursor.fetchall()]
        return {"total_records": total, "transactions": rows}
    except sqlite3.Error as e:
        print(f"Database error during retrieval: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not retrieve transactions.")
    finally:
        if conn:
            conn.close()

@app.get("/summary/daily")
async def daily_summary(item_name: Optional[str] = Query(None), date_from: Optional[date] = Query(None), date_to: Optional[date] = Query(None)):
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        where, params = [], []
        if item_name:
            where.append("item_name = ?")
            params.append(item_name)
        if date_from:
            where.append("transaction_date >= ?")
            params.append(date_from.isoformat())
        if date_to:
            where.append("transaction_date <= ?")
            params.append(date_to.isoformat())
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        cursor.execute(f"""
            SELECT
              transaction_date,
              ROUND(SUM(quantity_kg), 3) AS total_qty_kg,
              ROUND(SUM((sale_rate - purchase_rate) * quantity_kg), 2) AS total_profit
            FROM transactions
            {where_sql}
            GROUP BY transaction_date
            ORDER BY transaction_date DESC
        """, params)
        rows = [dict(row) for row in cursor.fetchall()]
        return {"rows": rows}
    except sqlite3.Error as e:
        print(f"Database error during summary: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not compute summary.")
    finally:
        if conn:
            conn.close()

@app.get("/items")
async def list_items():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT item_name FROM transactions ORDER BY item_name ASC")
        items = [r[0] for r in cursor.fetchall()]
        return {"items": items}
    except sqlite3.Error as e:
        print(f"Database error during items list: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not list items.")
    finally:
        if conn:
            conn.close()

@app.delete("/transactions/{tx_id}", response_model=DeleteResponse)
async def delete_transaction(tx_id: int = FPath(..., ge=1), api_key: str = Depends(get_api_key)):
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Transaction not found")
        conn.commit()
        return {"deleted_id": tx_id, "message": "Transaction deleted"}
    except sqlite3.Error as e:
        print(f"Database error during delete: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not delete transaction.")
    finally:
        if conn:
            conn.close()
