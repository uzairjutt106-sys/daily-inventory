import sqlite3
from datetime import date
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends, Query, Path as FPath
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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
                purchase_rate REAL NOT NULL,
                sale_rate REAL NOT NULL,
                quantity_kg REAL NOT NULL,
                transaction_date TEXT NOT NULL
            );
        """)
        # helpful indexes
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
    purchase_rate: float = Field(..., gt=0, example=2200, description="Cost price per kg.")
    sale_rate: float = Field(..., gt=0, example=25000, description="Selling price per kg.")
    quantity_kg: float = Field(..., gt=0, example=5.0, description="Quantity of the item in kilograms.")
    # new: allow back-dating (optional)
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

# --------- UI serving (same-origin) ----------
base_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=base_dir), name="static")

@app.get("/", include_in_schema=False)
async def serve_ui():
    """
    Serve inventory.html from the same folder as this file.
    Falls back to a small message if the file isn't present.
    """
    html_path = base_dir / "inventory.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(
        f"""<!doctype html><meta charset='utf-8'>
        <h1>inventory.html not found</h1>
        <p>Expected at: <code>{html_path}</code></p>
        <p>Put your <strong>inventory.html</strong> next to this Python file or open
           <a href="/static/inventory.html">/static/inventory.html</a> if you placed it there.</p>""",
        status_code=200
    )

# favicon to avoid 404s
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    ico = base_dir / "favicon.ico"
    if ico.exists():
        return FileResponse(ico)
    # tiny 1x1 fallback
    return HTMLResponse("", status_code=204)

# Simple health endpoint (no auth)
@app.get("/health", include_in_schema=False)
async def health():
    return {"ok": True}

# --------- API Endpoints ----------

@app.post("/transactions", status_code=201)
async def record_transaction(
    transaction: Transaction,
    api_key: str = Depends(get_api_key)
):
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
        cursor.execute(
            """
            INSERT INTO transactions (item_name, purchase_rate, sale_rate, quantity_kg, transaction_date)
            VALUES (?, ?, ?, ?, ?)
            """,
            data
        )
        conn.commit()
        unit_profit = transaction.sale_rate - transaction.purchase_rate
        total_profit = unit_profit * transaction.quantity_kg
        return {
            "message": "Transaction recorded successfully",
            "item_name": transaction.item_name,
            "quantity_kg": transaction.quantity_kg,
            "date": tx_date,
            "total_profit": round(total_profit, 2)
        }
    except sqlite3.Error as e:
        print(f"Database error during insertion: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not record transaction.")
    finally:
        if conn:
            conn.close()

@app.get("/transactions")
async def get_all_transactions(
    item_name: Optional[str] = Query(None, description="Filter by exact item name"),
    date_from: Optional[date] = Query(None, description="Inclusive, e.g. 2025-10-01"),
    date_to: Optional[date] = Query(None, description="Inclusive, e.g. 2025-10-31"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        where = []
        params = []
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
        # total
        cursor.execute(f"SELECT COUNT(*) AS c FROM transactions {where_sql}", params)
        total = cursor.fetchone()["c"]

        # page
        cursor.execute(
            f"""
            SELECT id, item_name, purchase_rate, sale_rate, quantity_kg, transaction_date
            FROM transactions
            {where_sql}
            ORDER BY transaction_date DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset]
        )
        rows = [dict(row) for row in cursor.fetchall()]
        return {"total_records": total, "transactions": rows}
    except sqlite3.Error as e:
        print(f"Database error during retrieval: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not retrieve transactions.")
    finally:
        if conn:
            conn.close()

@app.get("/summary/daily")
async def daily_summary(
    item_name: Optional[str] = Query(None),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None)
):
    """
    Returns rows like:
    { transaction_date, total_qty_kg, total_profit }
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        where = []
        params = []
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
        cursor.execute(
            f"""
            SELECT
              transaction_date,
              ROUND(SUM(quantity_kg), 3) AS total_qty_kg,
              ROUND(SUM((sale_rate - purchase_rate) * quantity_kg), 2) AS total_profit
            FROM transactions
            {where_sql}
            GROUP BY transaction_date
            ORDER BY transaction_date DESC
            """
            , params
        )
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
    """Distinct item names (useful for dropdowns)."""
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
async def delete_transaction(
    tx_id: int = FPath(..., ge=1),
    api_key: str = Depends(get_api_key)
):
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
