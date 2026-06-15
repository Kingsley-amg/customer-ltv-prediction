WITH cal AS (   -- calibration window: behaviour we learn from
    SELECT * FROM transactions WHERE invoice_date < '2011-09-09'
),
pred AS (       -- prediction window: the value we want to forecast
    SELECT customer_id, SUM(revenue) AS future_revenue
    FROM transactions
    WHERE invoice_date >= '2011-09-09'
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    CAST(julianday('2011-09-09') - julianday(MAX(c.invoice_date)) AS INT) AS recency_days,
    CAST(julianday('2011-09-09') - julianday(MIN(c.invoice_date)) AS INT) AS tenure_days,
    COUNT(DISTINCT c.invoice)                         AS frequency,
    ROUND(SUM(c.revenue), 2)                          AS monetary_total,
    ROUND(SUM(c.revenue) * 1.0 / COUNT(DISTINCT c.invoice), 2) AS avg_order_value,
    SUM(c.quantity)                                   AS total_items,
    COUNT(DISTINCT c.stock_code)                      AS distinct_products,
    COUNT(DISTINCT strftime('%Y-%m', c.invoice_date)) AS active_months,
    ROUND(SUM(CASE WHEN c.invoice_date >= '2011-06-11' THEN c.revenue ELSE 0 END), 2)
                                                      AS recent_revenue_90d,
    COALESCE(ROUND(p.future_revenue, 2), 0)           AS future_revenue
FROM cal c
LEFT JOIN pred p ON c.customer_id = p.customer_id
GROUP BY c.customer_id;
