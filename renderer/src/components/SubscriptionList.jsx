import React from "react";

const formatCurrency = (value, currency) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2
  }).format(value);

export default function SubscriptionList({
  subscriptions,
  currency,
  onEdit,
  simulation,
  onToggleSimulation
}) {
  return (
    <section className="card subscriptions">
      <header>
        <div>
          <h2>Subscriptions</h2>
          <p className="subtext">Tap to simulate cancellations or edit details.</p>
        </div>
        <div className="search">
          <input type="text" placeholder="Search by name or category" />
        </div>
      </header>
      <div className="subscription-list">
        {subscriptions.map((sub) => (
          <div
            key={sub.id}
            className={`subscription-row ${
              simulation.includes(sub.id) ? "muted" : ""
            }`}
          >
            <div className="badge" />
            <div className="details">
              <div>
                <strong>{sub.name}</strong>
                <span className="tag">{sub.category}</span>
              </div>
              <small>Next billing {sub.nextBillingDate}</small>
            </div>
            <div className="amount">
              {formatCurrency(sub.cost, currency)}
              <span>/{sub.frequency === "yearly" ? "yr" : "mo"}</span>
            </div>
            <div className="actions">
              <button onClick={() => onToggleSimulation(sub.id)}>
                {simulation.includes(sub.id) ? "Undo" : "Cancel?"}
              </button>
              <button className="ghost" onClick={() => onEdit(sub)}>
                Edit
              </button>
            </div>
          </div>
        ))}
        {subscriptions.length === 0 && (
          <div className="empty">
            <h3>No subscriptions yet</h3>
            <p>Add your first service to see insights instantly.</p>
          </div>
        )}
      </div>
    </section>
  );
}
