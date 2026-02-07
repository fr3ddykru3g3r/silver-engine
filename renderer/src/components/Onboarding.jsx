import React from "react";

export default function Onboarding({ onSample, onBlank }) {
  return (
    <div className="onboarding">
      <div className="onboarding-card">
        <p className="eyebrow">Welcome to Silver Engine</p>
        <h1>Make subscriptions feel effortless.</h1>
        <p>
          Track every service, understand your true monthly spend, and stay
          motivated with tiny wins.
        </p>
        <div className="onboarding-actions">
          <button className="primary" onClick={onSample}>
            Start with sample data
          </button>
          <button className="ghost" onClick={onBlank}>
            Start from scratch
          </button>
        </div>
        <div className="onboarding-steps">
          <div>
            <strong>1</strong>
            <span>Add your first subscription</span>
          </div>
          <div>
            <strong>2</strong>
            <span>Simulate cancellations instantly</span>
          </div>
          <div>
            <strong>3</strong>
            <span>Stay under budget and earn streaks</span>
          </div>
        </div>
      </div>
    </div>
  );
}
