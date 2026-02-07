import React from "react";

const formatCurrency = (value, currency) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2
  }).format(value);

export default function SummaryHeader({
  totals,
  simulatedTotals,
  budget,
  currency,
  onAdd
}) {
  const headroom = budget - totals.monthly;
  const simulatedSavings = totals.monthly - simulatedTotals.monthly;

  return (
    <section className="summary-header">
      <div>
        <p className="eyebrow">Today’s Spend</p>
        <h1>{formatCurrency(totals.monthly / 30, currency)}</h1>
        <p className="subtext">Monthly total {formatCurrency(totals.monthly, currency)}</p>
      </div>
      <div className="summary-metrics">
        <div>
          <p>Total annual</p>
          <strong>{formatCurrency(totals.annual, currency)}</strong>
        </div>
        <div>
          <p>Budget headroom</p>
          <strong className={headroom >= 0 ? "positive" : "negative"}>
            {formatCurrency(headroom, currency)}
          </strong>
        </div>
        <div>
          <p>Simulated savings</p>
          <strong>{formatCurrency(simulatedSavings, currency)}</strong>
        </div>
      </div>
      <div className="summary-actions">
        <button className="primary" onClick={onAdd}>
          + Add subscription
        </button>
        <button className="ghost">Check updates</button>
      </div>
    </section>
  );
}
