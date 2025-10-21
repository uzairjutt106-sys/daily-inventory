# Daily Inventory API Documentation

## Base URL

```
https://nontranscriptive-postmillennial-angelique.ngrok-free.dev
```

## Authentication
Most endpoints require an API key in the header:

```
X-API-Key: @uzair143
```

---

## Endpoints

### 1. Health Check
**GET `/health`**
- **Description:** Simple health check (no authentication required).
- **Response:**
  ```json
  { "ok": true }
  ```

---

### 2. Record Transaction
**POST `/transactions`**
- **Description:** Record a new purchase/sale transaction.
- **Headers:**  
  `X-API-Key: @uzair143` (required)
- **Request Body:**  
  ```json
  {
    "item_name": "alumnium",
    "purchase_rate": 400.0,
    "sale_rate": 500.0,
    "quantity_kg": 15.0,
    "transaction_date": "2025-10-17" // optional, defaults to today
  }
  ```
- **Response:**  
  ```json
  {
    "message": "Transaction recorded successfully",
    "item_name": "aluminum",
    "quantity_kg": 15.0,
    "date": "2025-10-17",
    "total_profit": 1500.0
  }
  ```

---

### 3. List Transactions
**GET `/transactions`**
- **Description:** List transactions with optional filters and pagination.
- **Query Parameters:**
  - `item_name` (string, optional): Filter by item name.
  - `date_from` (YYYY-MM-DD, optional): Start date (inclusive).
  - `date_to` (YYYY-MM-DD, optional): End date (inclusive).
  - `limit` (int, default 100): Max records to return (1-1000).
  - `offset` (int, default 0): Records to skip.
- **Response:**
  ```json
  {
    "total_records": 2,
    "transactions": [
      {
        "id": 1,
        "item_name": "plastic",
        "purchase_rate": 8.5,
        "sale_rate": 12.0,
        "quantity_kg": 50.0,
        "transaction_date": "2025-10-17"
      },
      // ...more transactions
    ]
  }
  ```

---

### 4. Daily Summary
**GET `/summary/daily`**
- **Description:** Get daily summary of quantities and profit.
- **Query Parameters:**
  - `item_name` (string, optional): Filter by item name.
  - `date_from` (YYYY-MM-DD, optional): Start date.
  - `date_to` (YYYY-MM-DD, optional): End date.
- **Response:**
  ```json
  {
    "rows": [
      {
        "transaction_date": "2025-10-17",
        "total_qty_kg": 50.0,
        "total_profit": 175.0
      },
      // ...more days
    ]
  }
  ```

---

### 5. List Items
**GET `/items`**
- **Description:** Get distinct item names (for dropdowns, etc).
- **Response:**
  ```json
  {
    "items": ["pitel", "copper", "aluminum"]
  }
  ```

---

### 6. Delete Transaction
**DELETE `/transactions/{tx_id}`**
- **Description:** Delete a transaction by its ID.
- **Headers:**  
  `X-API-Key: @uzair143` (required)
- **Path Parameter:**
  - `tx_id` (int): Transaction ID to delete.
- **Response:**
  ```json
  {
    "deleted_id": 1,
    "message": "Transaction deleted"
  }
  ```

---

## Error Handling
- All errors return JSON with a `detail` field describing the error.
- Common errors:
  - 400: Missing API key
  - 401: Invalid API key
  - 404: Transaction not found
  - 500: Database error

---

## Notes
- Dates must be in `YYYY-MM-DD` format.
- All quantities are in kilograms.
- All rates are per kilogram.
- For UI, you can serve `inventory.html` at `/` or `/static/inventory.html`.

---

For further details, see the FastAPI backend code in `main.py`.
