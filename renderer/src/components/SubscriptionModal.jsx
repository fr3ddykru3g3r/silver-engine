import React, { useMemo, useState } from "react";
import { normalizeMonthlyCost, normalizeAnnualCost } from "../../../core/subscriptions.js";

const emptyState = {
  id: "",
  name: "",
  category: "",
  frequency: "monthly",
  cost: 0,
  customInterval: 1,
  nextBillingDate: "",
  startDate: "",
  status: "active",
  notes: ""
};

const formatCurrency = (value, currency) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2
  }).format(value);

export default function SubscriptionModal({ initial, currency, onClose, onSave }) {
  const [form, setForm] = useState(
    initial ? { ...initial } : { ...emptyState, id: crypto.randomUUID() }
  );

  const monthly = useMemo(() => normalizeMonthlyCost(form), [form]);
  const annual = useMemo(() => normalizeAnnualCost(form), [form]);

  const update = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div className="modal-overlay">
      <div className="modal">
        <header>
          <div>
            <h2>{initial ? "Edit subscription" : "Add subscription"}</h2>
            <p className="subtext">Keep it simple. Add details later.</p>
          </div>
          <button className="ghost" onClick={onClose}>
            Close
          </button>
        </header>
        <div className="modal-body">
          <div className="field">
            <label>Name</label>
            <input
              value={form.name}
              onChange={(event) => update("name", event.target.value)}
              placeholder="Spotify Premium"
            />
          </div>
          <div className="field">
            <label>Category</label>
            <input
              value={form.category}
              onChange={(event) => update("category", event.target.value)}
              placeholder="Streaming"
            />
          </div>
          <div className="field">
            <label>Billing frequency</label>
            <select
              value={form.frequency}
              onChange={(event) => update("frequency", event.target.value)}
            >
              <option value="monthly">Monthly</option>
              <option value="yearly">Yearly</option>
              <option value="custom">Custom interval</option>
            </select>
          </div>
          {form.frequency === "custom" && (
            <div className="field">
              <label>Custom interval (months)</label>
              <input
                type="number"
                min="1"
                value={form.customInterval}
                onChange={(event) => update("customInterval", event.target.value)}
              />
            </div>
          )}
          <div className="field">
            <label>Cost</label>
            <input
              type="number"
              value={form.cost}
              onChange={(event) => update("cost", event.target.value)}
            />
          </div>
          <div className="field">
            <label>Next billing date</label>
            <input
              type="date"
              value={form.nextBillingDate}
              onChange={(event) => update("nextBillingDate", event.target.value)}
            />
          </div>
          <div className="field">
            <label>Notes</label>
            <textarea
              value={form.notes}
              onChange={(event) => update("notes", event.target.value)}
              placeholder="Add a reminder or renewal note"
            />
          </div>
          <div className="preview">
            <div>
              <span>Monthly impact</span>
              <strong>{formatCurrency(monthly, currency)}</strong>
            </div>
            <div>
              <span>Annual impact</span>
              <strong>{formatCurrency(annual, currency)}</strong>
            </div>
          </div>
        </div>
        <footer>
          <button className="primary" onClick={() => onSave(form)}>
            Save subscription
          </button>
        </footer>
      </div>
    </div>
  );
}
