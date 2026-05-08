export const normalizeMonthlyCost = (subscription) => {
  const amount = Number(subscription.cost || 0);
  if (subscription.frequency === "monthly") return amount;
  if (subscription.frequency === "yearly") return amount / 12;
  if (subscription.frequency === "custom") {
    const interval = Number(subscription.customInterval || 1);
    return interval === 0 ? 0 : amount / interval;
  }
  return amount;
};

export const normalizeAnnualCost = (subscription) => normalizeMonthlyCost(subscription) * 12;

export const computeTotals = (subscriptions = []) => {
  const monthly = subscriptions.reduce((sum, sub) => sum + normalizeMonthlyCost(sub), 0);
  const annual = monthly * 12;

  return {
    monthly,
    annual
  };
};

export const computeCategoryTotals = (subscriptions = []) => {
  return subscriptions.reduce((totals, sub) => {
    const key = sub.category || "Other";
    totals[key] = (totals[key] || 0) + normalizeMonthlyCost(sub);
    return totals;
  }, {});
};
