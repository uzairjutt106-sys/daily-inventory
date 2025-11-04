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
class SaleUpdate(BaseModel):
    item_name: str = Field(..., example="copper")
    sale_rate: float = Field(..., gt=0, description="Selling price per kg")
    quantity_kg: float = Field(..., gt=0, description="Quantity sold in kilograms")
    sale_date: Optional[date] = Field(None, description="Date of sale (YYYY-MM-DD)")

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
        "http://192.168.0.109:3000",
        "http://192.168.0.109:4000",
        "http://192.168.0.109:4001",
        "http://192.168.0.122:3000",
        "http://192.168.0.122:4000",
        "http://192.168.0.122:4001",
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
        'http://192.168.0.125:4001',
        'http://192.168.0.125:4000',
        'http://192.168.0.125:3001',
        'http://192.168.0.128:4000',
        'http://192.168.0.128:4001',
        'http://192.168.0.128:3000',
        'http://192.168.100.42:4000',
        'http://192.168.100.42:4001',
        'http://192.168.100.42:3000',
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

# --------- Helpers for sales profit computation ----------
def ensure_sales_table():
    """Create sales table if missing and make sure 'profit' column exists."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL,
            sale_rate REAL NOT NULL,
            quantity_kg REAL NOT NULL,
            sale_date TEXT NOT NULL,
            profit REAL
        );
    """)
    # 🔹 Helpful for per-item aggregation (for /stocks summary)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_item ON sales(item_name);")
    conn.commit()
    conn.close()


def get_weighted_avg_purchase(item_name: str) -> float:
    """Compute weighted-average purchase rate for an item from transactions."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            COALESCE(SUM(purchase_rate * quantity_kg), 0.0) AS total_cost,
            COALESCE(SUM(quantity_kg), 0.0) AS total_qty
        FROM transactions
        WHERE item_name = ?
          AND purchase_rate IS NOT NULL
    """, (item_name,))
    row = cursor.fetchone()
    conn.close()
    total_cost = row[0] or 0.0
    total_qty = row[1] or 0.0
    return (total_cost / total_qty) if total_qty > 0 else 0.0

def backfill_sales_profit():
    """Compute and fill missing profit values for existing sales rows."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # Find rows where profit is NULL
    cursor.execute("SELECT id, item_name, sale_rate, quantity_kg FROM sales WHERE profit IS NULL")
    rows = cursor.fetchall()
    updated = 0
    for sale_id, item_name, sale_rate, qty in rows:
        avg_purchase = get_weighted_avg_purchase(item_name)
        profit = (sale_rate - avg_purchase) * qty
        cursor.execute("UPDATE sales SET profit = ? WHERE id = ?", (profit, sale_id))
        updated += 1
    conn.commit()
    conn.close()
    if updated:
        print(f"Backfilled profit for {updated} sale(s).")

# Run sales table ensure + backfill once at import
ensure_sales_table()
backfill_sales_profit()
from datetime import datetime

def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def compute_profit_rows(granularity: str, start_date: str, end_date: str):
    """
    Returns a list of dict rows with: bucket_key, bucket_start, bucket_end, total_profit
    based on rows in the 'sales' table (uses 'profit' column).
    """
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Guard
    if granularity not in ("daily", "weekly", "monthly", "custom"):
        conn.close()
        raise ValueError("granularity must be one of: daily|weekly|monthly|custom")

    # Build SQL grouping for buckets
    if granularity == "daily":
        group_sql = """strftime('%Y-%m-%d', sale_date)"""
        sel_bucket = "strftime('%Y-%m-%d', sale_date) AS bucket_key"
        sel_start  = "strftime('%Y-%m-%d', sale_date) AS bucket_start"
        sel_end    = "strftime('%Y-%m-%d', sale_date) AS bucket_end"
    elif granularity == "weekly":
    # Compute Monday (week_start) and Sunday (week_end) for each sale_date,
    # group by the week_start, and label as YYYY-Www using week_start.
        cur.execute("""
        WITH f AS (
          SELECT date(sale_date) AS d, profit
          FROM sales
          WHERE date(sale_date) BETWEEN date(?) AND date(?)
        ),
        w AS (
          -- Monday of current week: date(d, 'weekday 1', '-7 days')
          SELECT
            date(d, 'weekday 1', '-7 days')                             AS week_start,
            date(date(d, 'weekday 1', '-7 days'), '+6 days')            AS week_end,
            profit
          FROM f
        )
        SELECT
          strftime('%Y', week_start) || '-W' ||
            printf('%02d', strftime('%W', week_start))                  AS bucket_key,
          week_start                                                    AS bucket_start,
          week_end                                                      AS bucket_end,
          ROUND(COALESCE(SUM(profit), 0), 2)                            AS total_profit
        FROM w
        GROUP BY bucket_key
        ORDER BY week_start ASC
    """, (start_date, end_date))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    elif granularity == "monthly":
        sel_bucket = "strftime('%Y-%m', sale_date) AS bucket_key"
        sel_start  = "date(strftime('%Y-%m-01', sale_date)) AS bucket_start"
        sel_end    = "date(strftime('%Y-%m-01', sale_date), '+1 month', '-1 day') AS bucket_end"
        group_sql  = "strftime('%Y-%m', sale_date)"
    else:  # custom => single bucket: whole range as one line
        cur.execute("""
            SELECT
              'custom' AS bucket_key,
              ?        AS bucket_start,
              ?        AS bucket_end,
              ROUND(COALESCE(SUM(profit), 0), 2) AS total_profit
            FROM sales
            WHERE date(sale_date) BETWEEN date(?) AND date(?)
        """, (start_date, end_date, start_date, end_date))
        rows = [dict(cur.fetchone())]
        conn.close()
        return rows

    # Non-custom: group by bucket within the range
    cur.execute(f"""
        SELECT
          {sel_bucket},
          {sel_start},
          {sel_end},
          ROUND(COALESCE(SUM(profit), 0), 2) AS total_profit
        FROM sales
        WHERE date(sale_date) BETWEEN date(?) AND date(?)
        GROUP BY {group_sql}
        ORDER BY bucket_key ASC
    """, (start_date, end_date))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


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

        # Profit is only meaningful on sales; keep return the same as before
        total_profit = None
        if transaction.sale_rate is not None and transaction.purchase_rate is not None:
            unit_profit = transaction.sale_rate - transaction.purchase_rate
            total_profit = unit_profit * transaction.quantity_kg

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
from datetime import datetime, timedelta

@app.get("/transactions/daily")
async def transactions_by_day(
    date_str: Optional[str] = Query(None, description="YYYY-MM-DD"),
    limit: int = Query(1000, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    api_key: str = Depends(get_api_key),
):
    """
    Returns all transactions for a single calendar date (local DB date),
    plus simple totals for that day.
    """
    # default to today if not provided
    the_day = date_str or date.today().isoformat()

    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # If your column stores plain 'YYYY-MM-DD', equality is perfect.
        # If you ever stored datetimes, you can switch to BETWEEN date() and date()+1 day - 1 sec.
        cur.execute("""
            SELECT id, item_name, purchase_rate, sale_rate, quantity_kg, transaction_date
            FROM transactions
            WHERE transaction_date = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (the_day, limit, offset))
        rows = [dict(r) for r in cur.fetchall()]

        # Totals for that day (helpful for footer UI)
        cur.execute("""
            SELECT 
              ROUND(SUM(quantity_kg), 3)                       AS total_qty,
              ROUND(SUM(COALESCE(purchase_rate,0)*quantity_kg), 2) AS total_amount
            FROM transactions
            WHERE transaction_date = ?
        """, (the_day,))
        totals = dict(cur.fetchone() or {"total_qty": 0, "total_amount": 0})

        return {
            "date": the_day,
            "rows": rows,
            "totals": totals,
            "count": len(rows),
            "limit": limit,
            "offset": offset
        }
    except sqlite3.Error as e:
        print("DB error /transactions/daily:", e)
        raise HTTPException(status_code=500, detail="Database error: could not load daily transactions.")
    finally:
        conn.close()


# ✅ Update an existing transaction (moved to proper top-level)
class TransactionUpdate(BaseModel):
    item_name: str = Field(..., example="copper")
    purchase_rate: Optional[float] = Field(None, ge=0, description="Cost per kg (can be null)")
    quantity_kg: float = Field(..., gt=0, description="Quantity in kg")
    transaction_date: Optional[date] = Field(None, description="YYYY-MM-DD")

@app.put("/transactions/{tx_id}")
async def update_transaction(
    tx_id: int = FPath(..., ge=1),
    payload: TransactionUpdate = ...,
    api_key: str = Depends(get_api_key),
):
    """
    Update a transaction row. Keeps existing sale_rate unchanged.
    Frontend sends: item_name, purchase_rate (int), quantity_kg, transaction_date (YYYY-MM-DD).
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Ensure record exists
        cur.execute("SELECT id, transaction_date FROM transactions WHERE id = ?", (tx_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Transaction not found")

        tx_date = (payload.transaction_date or date.fromisoformat(row["transaction_date"])).isoformat()

        cur.execute(
            """
            UPDATE transactions
               SET item_name = ?,
                   purchase_rate = ?,
                   quantity_kg = ?,
                   transaction_date = ?
             WHERE id = ?
            """,
            (
                payload.item_name.strip(),
                payload.purchase_rate,
                float(payload.quantity_kg),
                tx_date,
                tx_id,
            ),
        )

        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Transaction not found")

        conn.commit()

        # Return updated record
        cur.execute(
            "SELECT id, item_name, purchase_rate, sale_rate, quantity_kg, transaction_date FROM transactions WHERE id = ?",
            (tx_id,),
        )
        updated = dict(cur.fetchone())
        return {"message": "Transaction updated", "transaction": updated}

    except sqlite3.Error as e:
        print(f"DB error during update: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not update transaction.")
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
              ROUND(SUM((COALESCE(sale_rate,0) - COALESCE(purchase_rate,0)) * quantity_kg), 2) AS total_profit
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

# ---------- SALES ENDPOINTS ----------

class SaleIn(BaseModel):
    item_name: str = Field(..., example="copper")
    sale_rate: int = Field(..., gt=0, description="Selling price per kg (integer).")
    quantity_kg: float = Field(..., gt=0, description="Quantity sold in kilograms.")
    sale_date: Optional[date] = Field(None, description="Date of sale (defaults to today).")

ensure_sales_table()
backfill_sales_profit()

@app.post("/sales", status_code=201)
async def create_sale(sale: SaleIn, api_key: str = Depends(get_api_key)):
    """Record a new sale entry with computed profit saved to DB."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        sale_date = (sale.sale_date or date.today()).isoformat()

        # Compute weighted average purchase for this item
        avg_purchase = get_weighted_avg_purchase(sale.item_name)
        profit = (float(sale.sale_rate) - avg_purchase) * float(sale.quantity_kg)

        cursor.execute("""
            INSERT INTO sales (item_name, sale_rate, quantity_kg, sale_date, profit)
            VALUES (?, ?, ?, ?, ?)
        """, (sale.item_name, sale.sale_rate, sale.quantity_kg, sale_date, profit))
        conn.commit()

        return {
            "message": "Sale recorded successfully",
            "item_name": sale.item_name,
            "sale_rate": sale.sale_rate,
            "quantity_kg": sale.quantity_kg,
            "sale_date": sale_date,
            "profit": round(profit, 2),
        }
    except sqlite3.Error as e:
        print(f"Database error during sale insertion: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not record sale.")
    finally:
        if conn:
            conn.close()
@app.put("/sales/{sale_id}")
async def update_sale(
    sale_id: int = FPath(..., ge=1),
    payload: SaleUpdate = ...,
    api_key: str = Depends(get_api_key),
):
    """
    Update an existing sale and re-compute its profit using current weighted-average purchase.
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Ensure the sale exists
        cur.execute("SELECT id, item_name, sale_date FROM sales WHERE id = ?", (sale_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Sale not found")

        sale_dt = (payload.sale_date or date.fromisoformat(row["sale_date"])).isoformat()

        # Recompute profit using current weighted average purchase for *new* item_name
        avg_purchase = get_weighted_avg_purchase(payload.item_name)
        profit = (float(payload.sale_rate) - avg_purchase) * float(payload.quantity_kg)

        # Update
        cur.execute("""
            UPDATE sales
               SET item_name   = ?,
                   sale_rate   = ?,
                   quantity_kg = ?,
                   sale_date   = ?,
                   profit      = ?
             WHERE id = ?
        """, (
            payload.item_name.strip(),
            float(payload.sale_rate),
            float(payload.quantity_kg),
            sale_dt,
            float(profit),
            sale_id,
        ))

        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Sale not found")

        conn.commit()

        # Return updated record
        cur.execute("SELECT id, item_name, sale_rate, quantity_kg, sale_date, profit FROM sales WHERE id = ?", (sale_id,))
        updated = dict(cur.fetchone())
        return {"message": "Sale updated", "sale": updated}

    except sqlite3.Error as e:
        print(f"DB error during sale update: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not update sale.")
    finally:
        if conn:
            conn.close()


# --------- Profit storage (materialized) ----------
def ensure_profit_tables():
    """
    Stores aggregated profit rows you generate.
    One row per bucket (daily/weekly/monthly/custom).
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profit_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            granularity TEXT NOT NULL,            -- 'daily' | 'weekly' | 'monthly' | 'custom'
            bucket_key TEXT NOT NULL,             -- e.g. '2025-10-26' (daily), '2025-W43' (weekly), '2025-10' (monthly)
            start_date TEXT NOT NULL,             -- bucket start
            end_date   TEXT NOT NULL,             -- bucket end (inclusive)
            total_profit REAL NOT NULL DEFAULT 0, -- sum of sales.profit in bucket
            created_at TEXT NOT NULL              -- when this report row was generated
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_profit_reports_lookup ON profit_reports(granularity, start_date, end_date, bucket_key)")
    conn.commit()
    conn.close()

ensure_profit_tables()


def create_stocks_view():
    """Creates a SQL view 'stocks_view' that shows per-item net stock and profit."""
    conn = sqlite3.connect(DATABASE_FILE)
    cur = conn.cursor()

    cur.execute("DROP VIEW IF EXISTS stocks_view;")
    cur.execute("""
        CREATE VIEW IF NOT EXISTS stocks_view AS
        WITH items AS (
            SELECT item_name FROM transactions
            UNION
            SELECT item_name FROM sales
        ),
        p AS (
            SELECT item_name, SUM(quantity_kg) AS purchased
            FROM transactions
            GROUP BY item_name
        ),
        s AS (
            SELECT item_name,
                   SUM(quantity_kg) AS sold,
                   SUM(COALESCE(profit, 0)) AS realized_profit
            FROM sales
            GROUP BY item_name
        )
        SELECT
            i.item_name AS item_name,
            ROUND(COALESCE(p.purchased, 0), 3) AS purchased_kg,
            ROUND(COALESCE(s.sold, 0), 3) AS sold_kg,
            ROUND(COALESCE(p.purchased, 0) - COALESCE(s.sold, 0), 3) AS net_kg,
            ROUND(COALESCE(s.realized_profit, 0), 2) AS realized_profit
        FROM items i
        LEFT JOIN p ON p.item_name = i.item_name
        LEFT JOIN s ON s.item_name = i.item_name
        ORDER BY i.item_name COLLATE NOCASE ASC;
    """)
    conn.commit()
    conn.close()
    print("✅ stocks_view created successfully.")
create_stocks_view()         
from pydantic import BaseModel

class ProfitGenerateIn(BaseModel):
    start_date: str  # 'YYYY-MM-DD'
    end_date: str    # 'YYYY-MM-DD'
    granularity: str # 'daily' | 'weekly' | 'monthly' | 'custom'

@app.post("/profits/generate", status_code=201)
async def generate_profits(body: ProfitGenerateIn, api_key: str = Depends(get_api_key)):
    """
    Computes profits from sales.profit for the given range & granularity,
    then INSERTs rows into 'profit_reports'. Returns the inserted rows.
    """
    start_date = body.start_date
    end_date   = body.end_date
    gran       = body.granularity

    # Compute rows
    rows = compute_profit_rows(gran, start_date, end_date)

    # Save into table
    conn = sqlite3.connect(DATABASE_FILE)
    cur = conn.cursor()

    # Optional: remove overlapping existing rows for same granularity & date window
    cur.execute("""
    DELETE FROM profit_reports
     WHERE granularity = ?
       AND date(end_date) >= date(?)   -- bucket ends on/after requested start
       AND date(start_date) <= date(?) -- bucket starts on/before requested end
""", (gran, start_date, end_date))


    now = date.today().isoformat()
    for r in rows:
        cur.execute("""
            INSERT INTO profit_reports (granularity, bucket_key, start_date, end_date, total_profit, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            gran,
            r["bucket_key"],
            r["bucket_start"],
            r["bucket_end"],
            float(r["total_profit"] or 0),
            now
        ))
    conn.commit()

    # Return what we just stored
    cur.execute("""
        SELECT id, granularity, bucket_key, start_date, end_date, total_profit, created_at
        FROM profit_reports
        WHERE granularity = ?
          AND date(start_date) >= date(?)
          AND date(end_date)   <= date(?)
        ORDER BY bucket_key ASC
    """, (gran, start_date, end_date))
    out = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]
    conn.close()

    return {"inserted": out, "count": len(out)}

@app.get("/sales")
async def list_sales(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    """List all recorded sales (now includes profit)."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, item_name, sale_rate, quantity_kg, sale_date, profit
            FROM sales
            ORDER BY sale_date DESC, id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        rows = [dict(row) for row in cursor.fetchall()]
        return {"sales": rows, "count": len(rows)}
    except sqlite3.Error as e:
        print(f"Database error during sales list: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not retrieve sales.")
    finally:
        if conn:
            conn.close()

@app.delete("/sales/{sale_id}", response_model=DeleteResponse)
async def delete_sale(sale_id: int = FPath(..., ge=1), api_key: str = Depends(get_api_key)):
    """Delete a sale record by ID."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sales WHERE id = ?", (sale_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Sale not found")
        conn.commit()
        return {"deleted_id": sale_id, "message": "Sale deleted"}
    except sqlite3.Error as e:
        print(f"Database error during sale delete: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not delete sale.")
    finally:
        if conn:
            conn.close()

@app.get("/stocks")
async def stocks_summary():
    """
    Returns per-item stock summary:
      - purchased_kg: total purchases
      - sold_kg: total sales
      - net_kg: purchased - sold
      - realized_profit: total profit (sum of sales.profit)
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            WITH items AS (
              SELECT item_name FROM transactions
              UNION
              SELECT item_name FROM sales
            ),
            p AS (
              SELECT item_name, SUM(quantity_kg) AS purchased
              FROM transactions
              GROUP BY item_name
            ),
            s AS (
              SELECT item_name,
                     SUM(quantity_kg) AS sold,
                     SUM(COALESCE(profit, 0)) AS realized_profit
              FROM sales
              GROUP BY item_name
            )
            SELECT
              i.item_name AS item_name,
              ROUND(COALESCE(p.purchased, 0), 3) AS purchased_kg,
              ROUND(COALESCE(s.sold, 0), 3) AS sold_kg,
              ROUND(COALESCE(p.purchased, 0) - COALESCE(s.sold, 0), 3) AS net_kg,
              ROUND(COALESCE(s.realized_profit, 0), 2) AS realized_profit
            FROM items i
            LEFT JOIN p ON p.item_name = i.item_name
            LEFT JOIN s ON s.item_name = i.item_name
            ORDER BY i.item_name COLLATE NOCASE ASC;
        """)
        rows = [dict(r) for r in cur.fetchall()]
        return {"stocks": rows, "count": len(rows)}
    except sqlite3.Error as e:
        print(f"DB error during stocks summary: {e}")
        raise HTTPException(status_code=500, detail="Database error: Could not compute stocks.")
    finally:
        if conn:
            conn.close()
@app.get("/profits")
async def list_profits(
    granularity: Optional[str] = Query(None, description="daily|weekly|monthly|custom"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str]   = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where = []
    args  = []

    if granularity:
        where.append("granularity = ?")
        args.append(granularity)

    # Overlap logic for ranges:
    # include buckets where (bucket_end >= start) AND (bucket_start <= end)
    if start_date:
        where.append("date(end_date) >= date(?)")
        args.append(start_date)
    if end_date:
        where.append("date(start_date) <= date(?)")
        args.append(end_date)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    cur.execute(f"""
        SELECT id, granularity, bucket_key, start_date, end_date, total_profit, created_at
        FROM profit_reports
        {where_sql}
        ORDER BY bucket_key ASC
        LIMIT ? OFFSET ?
    """, args + [limit, offset])

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"profits": rows, "count": len(rows)}



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