import React from "react";

const formatCurrency = (value, currency) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2
  }).format(value);

export default function InsightsPanel({ categories, totals, simulatedTotals, currency }) {
  const entries = Object.entries(categories);
  const remaining = totals.monthly - simulatedTotals.monthly;

  return (
    <aside className="card insights">
      <h2>Insights</h2>
      <div className="insight-block">
        <h3>What-if savings</h3>
        <p>
          You’d save <strong>{formatCurrency(remaining, currency)}</strong> monthly if
          you cancel the selected subscriptions.
        </p>
      </div>
      <div className="insight-block">
        <h3>Spend by category</h3>
        <div className="category-list">
          {entries.map(([name, value]) => (
            <div key={name} className="category-row">
              <span>{name}</span>
              <strong>{formatCurrency(value, currency)}</strong>
            </div>
          ))}
        </div>
      </div>
      <div className="insight-block">
        <h3>Streak</h3>
        <p className="streak">You’ve stayed under budget for 4 months 🎉</p>
      </div>
      <div className="insight-block">
        <h3>Upcoming renewals</h3>
        <ul>
          <li>Netflix · 2 days</li>
          <li>Notion · 12 days</li>
          <li>Mobile Plan · 18 days</li>
        </ul>
      </div>
    </aside>
  );
}
