import React, { useMemo, useState } from "react";
import SummaryHeader from "./components/SummaryHeader.jsx";
import SubscriptionList from "./components/SubscriptionList.jsx";
import SubscriptionModal from "./components/SubscriptionModal.jsx";
import InsightsPanel from "./components/InsightsPanel.jsx";
import Onboarding from "./components/Onboarding.jsx";
import { computeTotals, computeCategoryTotals } from "../../core/subscriptions.js";

const sampleSubscriptions = [
  {
    id: "netflix",
    name: "Netflix",
    category: "Streaming",
    frequency: "monthly",
    cost: 15.99,
    nextBillingDate: "2024-09-24",
    status: "active"
  },
  {
    id: "notion",
    name: "Notion Plus",
    category: "Productivity",
    frequency: "yearly",
    cost: 96,
    nextBillingDate: "2025-01-02",
    status: "active"
  },
  {
    id: "phone",
    name: "Mobile Plan",
    category: "Utilities",
    frequency: "monthly",
    cost: 35,
    nextBillingDate: "2024-09-14",
    status: "active"
  }
];

const defaultPreferences = {
  currency: "USD",
  theme: "dark",
  monthlyBudget: 200
};

const loadInitialData = async () => {
  if (window.subscriptionAPI) {
    const [subscriptions, preferences] = await Promise.all([
      window.subscriptionAPI.list(),
      window.subscriptionAPI.loadPreferences()
    ]);
    return {
      subscriptions,
      preferences
    };
  }
  return { subscriptions: [], preferences: defaultPreferences };
};

export default function App() {
  const [subscriptions, setSubscriptions] = useState([]);
  const [preferences, setPreferences] = useState(defaultPreferences);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [editing, setEditing] = useState(null);
  const [simulation, setSimulation] = useState([]);
  const [showOnboarding, setShowOnboarding] = useState(true);

  React.useEffect(() => {
    let mounted = true;
    loadInitialData().then((data) => {
      if (!mounted) return;
      setSubscriptions(data.subscriptions);
      setPreferences(data.preferences || defaultPreferences);
      setShowOnboarding(data.subscriptions.length === 0);
      setLoading(false);
    });
    return () => {
      mounted = false;
    };
  }, []);

  const totals = useMemo(() => computeTotals(subscriptions), [subscriptions]);
  const categories = useMemo(() => computeCategoryTotals(subscriptions), [subscriptions]);

  const simulatedSubscriptions = subscriptions.filter(
    (sub) => !simulation.includes(sub.id)
  );
  const simulatedTotals = useMemo(
    () => computeTotals(simulatedSubscriptions),
    [simulatedSubscriptions]
  );

  const handleSave = (payload) => {
    const next = editing
      ? subscriptions.map((sub) => (sub.id === editing.id ? payload : sub))
      : [...subscriptions, payload];

    setSubscriptions(next);
    window.subscriptionAPI?.save(next);
    setShowModal(false);
    setEditing(null);
  };

  const handleSample = () => {
    setSubscriptions(sampleSubscriptions);
    window.subscriptionAPI?.save(sampleSubscriptions);
    setShowOnboarding(false);
  };

  const handleBlank = () => {
    setSubscriptions([]);
    setShowOnboarding(false);
  };

  if (loading) {
    return <div className="loading">Preparing your dashboard…</div>;
  }

  return (
    <div className={`app theme-${preferences.theme}`}>
      {showOnboarding ? (
        <Onboarding onSample={handleSample} onBlank={handleBlank} />
      ) : (
        <>
          <SummaryHeader
            totals={totals}
            simulatedTotals={simulatedTotals}
            budget={preferences.monthlyBudget}
            currency={preferences.currency}
            onAdd={() => setShowModal(true)}
            onDownload={() => window.subscriptionAPI?.exportPackage()}
          />
          <div className="main-grid">
            <SubscriptionList
              subscriptions={subscriptions}
              currency={preferences.currency}
              onEdit={(sub) => {
                setEditing(sub);
                setShowModal(true);
              }}
              simulation={simulation}
              onToggleSimulation={(id) => {
                setSimulation((prev) =>
                  prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
                );
              }}
            />
            <InsightsPanel
              categories={categories}
              totals={totals}
              simulatedTotals={simulatedTotals}
              currency={preferences.currency}
            />
          </div>
        </>
      )}
      {showModal && (
        <SubscriptionModal
          initial={editing}
          currency={preferences.currency}
          onClose={() => {
            setShowModal(false);
            setEditing(null);
          }}
          onSave={handleSave}
        />
      )}
    </div>
  );
}
