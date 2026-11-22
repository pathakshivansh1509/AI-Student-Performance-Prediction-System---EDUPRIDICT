document.addEventListener("DOMContentLoaded", () => {
    // ==============================
    // ELEMENTS
    // ==============================
    const predictionForm = document.getElementById("predictionForm");
    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultContent = document.getElementById("resultContent");
    const predictionScore = document.getElementById("predictionScore");
    const performanceCategory = document.getElementById("performanceCategory");

    const modelR2 = document.getElementById("modelR2");
    const modelMAE = document.getElementById("modelMAE");
    const modelRMSE = document.getElementById("modelRMSE");
    const aiInsight = document.getElementById("aiInsight");

    const historyList = document.getElementById("historyList");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");

    const predictButton = predictionForm
        ? predictionForm.querySelector("button[type='submit']")
        : null;

    const sidebar = document.querySelector(".sidebar");
    const menuButton = document.querySelector(".mobile-menu");
    const sidebarLinks = document.querySelectorAll(
        ".sidebar .nav-item"
    );

    // ==============================
    // ICONS
    // ==============================
    function refreshIcons() {
        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    // ==============================
    // SIDEBAR NAVIGATION FIX
    // ==============================
    sidebarLinks.forEach((link) => {
        link.addEventListener("click", (event) => {
            const href = link.getAttribute("href");

            if (!href || href === "#") {
                event.preventDefault();
                return;
            }

            if (href.startsWith("#")) {
                const target = document.getElementById(
                    href.substring(1)
                );

                if (!target) {
                    event.preventDefault();
                    return;
                }

                event.preventDefault();

                sidebarLinks.forEach((item) => {
                    item.classList.remove("active");
                });

                link.classList.add("active");

                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

                history.replaceState(null, "", href);
            }

            if (sidebar && window.innerWidth <= 900) {
                sidebar.classList.remove("active");
            }
        });
    });

    // ==============================
    // MOBILE MENU
    // ==============================
    if (sidebar && menuButton) {
        menuButton.addEventListener("click", () => {
            sidebar.classList.toggle("active");
        });
    }

    // ==============================
    // PREDICTION RESULT
    // ==============================
    function showPredictionResult() {
        if (resultPlaceholder) {
            resultPlaceholder.classList.add("hidden");
            resultPlaceholder.style.display = "none";
        }

        if (resultContent) {
            resultContent.classList.remove("hidden");
            resultContent.style.display = "block";
        }
    }

    function getDefaultInsight(category) {
        const insights = {
            POOR:
                "The prediction indicates that the student may need additional academic support.",

            AVERAGE:
                "Consistent study habits and regular attendance may help improve future performance.",

            GOOD:
                "The student shows a positive predicted performance trend.",

            EXCELLENT:
                "The student shows an excellent predicted performance. Maintaining consistent study habits can help."
        };

        return insights[category] ||
            "The prediction was generated using the trained Random Forest model.";
    }

    // ==============================
    // PREDICTION FORM
    // ==============================
    if (predictionForm) {
        predictionForm.addEventListener("submit", async (event) => {
            event.preventDefault();

            const fields = [
                ["study_hours", "studyHours"],
                ["attendance", "attendance"],
                ["previous_score", "previousScore"],
                ["assignment_score", "assignmentScore"],
                ["internal_score", "internalScore"],
                ["sleep_hours", "sleepHours"],
                ["backlogs", "backlogs"]
            ];

            const data = {};

            for (const [key, id] of fields) {
                const input = document.getElementById(id);
                const value = input.value.trim();

                if (value === "" || !Number.isFinite(Number(value))) {
                    alert("Please enter valid values in all fields.");
                    input.focus();
                    return;
                }

                data[key] = Number(value);
            }

            const originalButtonHTML = predictButton
                ? predictButton.innerHTML
                : "";

            if (predictButton) {
                predictButton.disabled = true;
                predictButton.textContent = "Predicting...";
            }

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (!response.ok || !result.success) {
                    throw new Error(
                        result.error || "Prediction failed."
                    );
                }

                const prediction = Number(result.prediction);
                const category = result.category || "AVERAGE";

                if (predictionScore) {
                    predictionScore.textContent =
                        prediction.toFixed(1);
                }

                if (performanceCategory) {
                    performanceCategory.textContent = category;

                    performanceCategory.classList.remove(
                        "poor",
                        "average",
                        "good",
                        "excellent"
                    );

                    performanceCategory.classList.add(
                        category.toLowerCase()
                    );
                }

                if (modelR2 && result.model_r2 != null) {
                    modelR2.textContent =
                        Number(result.model_r2).toFixed(2) + "%";
                }

                if (modelMAE && result.mae != null) {
                    modelMAE.textContent =
                        Number(result.mae).toFixed(2);
                }

                if (modelRMSE && result.rmse != null) {
                    modelRMSE.textContent =
                        Number(result.rmse).toFixed(2);
                }

                if (aiInsight) {
                    aiInsight.textContent =
                        result.insight || getDefaultInsight(category);
                }

                showPredictionResult();

                if (result.feature_importance) {
                    updateFeatureImportance(
                        result.feature_importance
                    );
                }

                await loadHistory();

            } catch (error) {
                console.error("Prediction error:", error);
                alert(error.message || "Unable to generate prediction.");

            } finally {
                if (predictButton) {
                    predictButton.disabled = false;
                    predictButton.innerHTML = originalButtonHTML;
                    refreshIcons();
                }
            }
        });
    }

    // ==============================
    // FEATURE IMPORTANCE
    // ==============================
    function formatFeatureName(name) {
        const names = {
            study_hours: "Study Hours",
            attendance: "Attendance",
            previous_score: "Previous Exam Score",
            assignment_score: "Assignment Score",
            internal_score: "Internal Score",
            sleep_hours: "Sleep Hours",
            backlogs: "Previous Backlogs"
        };

        return names[name] || String(name).replaceAll("_", " ");
    }

    function updateFeatureImportance(featureImportance) {
        const container = document.querySelector(".factor-grid");

        if (!container || !featureImportance) {
            return;
        }

        let entries = [];

        if (
            typeof featureImportance === "object" &&
            !Array.isArray(featureImportance)
        ) {
            entries = Object.entries(featureImportance);

        } else if (Array.isArray(featureImportance)) {
            entries = featureImportance.map((item) => {
                if (item && typeof item === "object") {
                    return [
                        item.feature || item.name || "Feature",
                        item.importance ?? item.value ?? 0
                    ];
                }

                return ["Feature", Number(item) || 0];
            });
        }

        entries.sort((a, b) => Number(b[1]) - Number(a[1]));

        container.innerHTML = entries.map(([feature, importance]) => {
            const value = Number(importance) || 0;
            const percentage = value <= 1 ? value * 100 : value;
            const safePercentage = Math.max(
                0,
                Math.min(percentage, 100)
            );

            return `
                <div class="factor">
                    <div class="factor-top">
                        <span>${formatFeatureName(feature)}</span>
                        <strong>${percentage.toFixed(2)}%</strong>
                    </div>
                    <div class="factor-bar">
                        <div style="width: ${safePercentage}%"></div>
                    </div>
                </div>
            `;
        }).join("");

        refreshIcons();
    }

    // ==============================
    // LOAD PREDICTION HISTORY
    // ==============================
    async function loadHistory() {
        if (!historyList) {
            return;
        }

        try {
            const response = await fetch("/history", {
                cache: "no-store"
            });

            if (!response.ok) {
                throw new Error("History request failed.");
            }

            const data = await response.json();

            if (!data.success) {
                throw new Error("Unable to retrieve history.");
            }

            renderHistory(data.history || []);

        } catch (error) {
            console.error("History loading error:", error);

            historyList.innerHTML = `
                <div class="history-empty">
                    <h3>Unable to load history</h3>
                    <p>Please refresh the page and try again.</p>
                </div>
            `;
        }
    }

    // ==============================
    // RENDER HISTORY
    // ==============================
    function renderHistory(historyItems) {
        if (!historyList) {
            return;
        }

        if (!historyItems.length) {
            historyList.innerHTML = `
                <div class="history-empty">
                    <div class="placeholder-icon">
                        <i data-lucide="clock-3"></i>
                    </div>
                    <h3>No predictions yet</h3>
                    <p>Make your first prediction and it will appear here.</p>
                </div>
            `;

            refreshIcons();
            return;
        }

        historyList.innerHTML = historyItems.map((item) => {
            const prediction = Number(item.prediction);
            const category = item.category || "AVERAGE";
            const categoryClass = category.toLowerCase();

            return `
                <div class="history-card">
                    <div class="history-main">
                        <div class="history-icon">
                            <i data-lucide="brain-circuit"></i>
                        </div>

                        <div class="history-info">
                            <div class="history-title">
                                Student Performance Prediction
                            </div>
                            <div class="history-time">
                                <i data-lucide="clock-3"></i>
                                ${item.timestamp || "Recent prediction"}
                            </div>
                        </div>

                        <div class="history-score">
                            <strong>
                                ${Number.isFinite(prediction)
                                    ? prediction.toFixed(1)
                                    : "—"}%
                            </strong>
                            <span class="${categoryClass}">
                                ${category}
                            </span>
                        </div>
                    </div>

                    <div class="history-details">
                        <div class="history-inputs">
                            <div class="history-input">
                                <span>Study Hours</span>
                                <strong>${item.study_hours ?? "—"}</strong>
                            </div>
                            <div class="history-input">
                                <span>Attendance</span>
                                <strong>${item.attendance ?? "—"}%</strong>
                            </div>
                            <div class="history-input">
                                <span>Previous Score</span>
                                <strong>${item.previous_score ?? "—"}</strong>
                            </div>
                            <div class="history-input">
                                <span>Assignment</span>
                                <strong>${item.assignment_score ?? "—"}</strong>
                            </div>
                            <div class="history-input">
                                <span>Internal Score</span>
                                <strong>${item.internal_score ?? "—"}</strong>
                            </div>
                            <div class="history-input">
                                <span>Sleep Hours</span>
                                <strong>${item.sleep_hours ?? "—"}</strong>
                            </div>
                            <div class="history-input">
                                <span>Backlogs</span>
                                <strong>${item.backlogs ?? "—"}</strong>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join("");

        refreshIcons();
    }

    // ==============================
    // CLEAR HISTORY
    // ==============================
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener("click", async () => {
            const confirmed = confirm(
                "Are you sure you want to clear all prediction history?"
            );

            if (!confirmed) {
                return;
            }

            clearHistoryBtn.disabled = true;

            try {
                const response = await fetch("/history/clear", {
                    method: "POST"
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(
                        data.error || "Unable to clear history."
                    );
                }

                await loadHistory();

            } catch (error) {
                console.error("Clear history error:", error);
                alert(error.message || "Unable to clear history.");

            } finally {
                clearHistoryBtn.disabled = false;
            }
        });
    }

    // ==============================
    // INITIAL LOAD
    // ==============================
    loadHistory();
    refreshIcons();

    console.log("EduPredict AI System loaded successfully.");
});