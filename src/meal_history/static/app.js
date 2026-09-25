/**
 * Frontend logic for the Meal History Dashboard.
 * Synchronized with validation checkpoints and 2025-2026 holiday calendar.
 */

let currentPage = 1;
const pageSize = 20;
let searchDebounceTimer = null;

document.addEventListener("DOMContentLoaded", () => {
    initDashboard();
    setupEventListeners();
});

async function initDashboard() {
    // Thiết lập ngày mặc định cho bộ chọn dự báo là ngày mai
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    const dateInput = document.getElementById("forecast-date");
    dateInput.value = tomorrow.toISOString().split("T")[0];

    // Kiểm tra và tải dữ liệu tuần tự
    const isReady = await checkHealthAndStatus();
    if (isReady) {
        await Promise.all([
            fetchValidationMetrics(),
            fetchSummaryKPIs(),
            fetchFilterOptions(),
            fetchForecastData(dateInput.value),
            fetchRecordsData()
        ]);
    }
}

function setupEventListeners() {
    document.getElementById("btn-forecast").addEventListener("click", () => {
        const selectedDate = document.getElementById("forecast-date").value;
        if (selectedDate) fetchForecastData(selectedDate);
    });

    document.getElementById("search-input").addEventListener("input", () => {
        clearTimeout(searchDebounceTimer);
        searchDebounceTimer = setTimeout(() => {
            currentPage = 1;
            fetchRecordsData();
        }, 350);
    });

    document.getElementById("department-select").addEventListener("change", () => {
        currentPage = 1;
        fetchRecordsData();
    });

    document.getElementById("status-select").addEventListener("change", () => {
        currentPage = 1;
        fetchRecordsData();
    });

    document.getElementById("btn-prev").addEventListener("click", () => {
        if (currentPage > 1) {
            currentPage--;
            fetchRecordsData();
        }
    });

    document.getElementById("btn-next").addEventListener("click", () => {
        currentPage++;
        fetchRecordsData();
    });
}

// 1. Kiểm tra trạng thái máy chủ & tính sẵn sàng của file
async function checkHealthAndStatus() {
    const connIndicator = document.getElementById("connection-status");
    const connText = document.getElementById("connection-text");
    const alertBanner = document.getElementById("service-alert");
    const alertMessage = document.getElementById("alert-message");

    try {
        const res = await fetch("/api/health");
        if (res.status === 503) {
            const errData = await res.json();
            showServiceAlert(errData.detail || "Dữ liệu chưa được khởi tạo. Vui lòng chạy pipeline cập nhật.");
            setIndicator(false, "Chưa khởi tạo dữ liệu");
            return false;
        }

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        setIndicator(true, "Hệ thống sẵn sàng");
        alertBanner.classList.add("hidden");
        return true;
    } catch (e) {
        setIndicator(false, "Mất kết nối API");
        showServiceAlert("Không thể kết nối đến máy chủ API backend.");
        return false;
    }
}

function setIndicator(online, text) {
    const connIndicator = document.getElementById("connection-status");
    const dot = connIndicator.querySelector(".dot");
    document.getElementById("connection-text").innerText = text;
    dot.className = `dot ${online ? "dot-online" : "dot-offline"}`;
}

function showServiceAlert(msg) {
    const alertBanner = document.getElementById("service-alert");
    document.getElementById("alert-message").innerText = msg;
    alertBanner.classList.remove("hidden");
}

// 2. Tải chỉ số kiểm chuẩn chất lượng (Data Validation)
async function fetchValidationMetrics() {
    try {
        const res = await fetch("/api/validation-status");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("val-total-rows").innerText = Number(data.total_rows).toLocaleString("vi-VN") + " dòng";

        // Kiểm tra tỷ lệ ngày hợp lệ (≥95%)
        const dateRate = data.date_validation.parse_rate_percent;
        const dateEl = document.getElementById("val-date-rate");
        const dateBadge = document.getElementById("val-date-badge");
        dateEl.innerText = `${dateRate}%`;

        if (data.date_validation.is_valid) {
            dateBadge.innerText = "Đạt chuẩn (≥95%)";
            dateBadge.className = "metric-badge badge-success";
        } else {
            dateBadge.innerText = "Cảnh báo (<95%)";
            dateBadge.className = "metric-badge badge-danger";
        }

        if (data.date_validation.min_date && data.date_validation.max_date) {
            document.getElementById("val-date-range").innerText = 
                `Từ ${data.date_validation.min_date} đến ${data.date_validation.max_date}`;
        }

        // Kiểm tra trạng thái phân loại nhị phân
        const statusCount = data.status_validation.unique_count;
        const statusEl = document.getElementById("val-status-count");
        const statusBadge = document.getElementById("val-status-badge");
        statusEl.innerText = `${statusCount} giá trị`;

        if (data.status_validation.is_binary) {
            statusBadge.innerText = "Chuẩn 2 trạng thái";
            statusBadge.className = "metric-badge badge-success";
        } else {
            statusBadge.innerText = "Sai lệch (>2 trạng thái)";
            statusBadge.className = "metric-badge badge-danger";
        }
    } catch (e) {
        console.error("Lỗi khi nạp dữ liệu kiểm chuẩn:", e);
    }
}

// 3. Tải thống kê tổng quan (Summary KPIs)
async function fetchSummaryKPIs() {
    try {
        const res = await fetch("/api/summary");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("sum-valid").innerText = Number(data.valid_rows).toLocaleString("vi-VN");
        document.getElementById("sum-invalid").innerText = Number(data.invalid_rows).toLocaleString("vi-VN");
        document.getElementById("sum-depts").innerText = data.departments;
        document.getElementById("sum-amount").innerText = 
            Number(data.total_amount).toLocaleString("vi-VN") + " đ";
    } catch (e) {
        console.error("Lỗi khi nạp tổng quan:", e);
    }
}

// 4. Tải bộ lọc Phòng ban & Trạng thái
async function fetchFilterOptions() {
    try {
        const res = await fetch("/api/options");
        if (!res.ok) return;
        const data = await res.json();

        const deptSelect = document.getElementById("department-select");
        (data.departments || []).forEach(d => {
            const opt = document.createElement("option");
            opt.value = d;
            opt.textContent = d;
            deptSelect.appendChild(opt);
        });

        const statusSelect = document.getElementById("status-select");
        (data.statuses || []).forEach(s => {
            const opt = document.createElement("option");
            opt.value = s;
            opt.textContent = s;
            statusSelect.appendChild(opt);
        });
    } catch (e) {
        console.error("Lỗi khi nạp options:", e);
    }
}

// 5. Chạy và hiển thị dự báo theo ngày
async function fetchForecastData(targetDate) {
    try {
        const res = await fetch(`/api/forecast?target_date=${targetDate}`);
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("forecast-target-label").innerText = 
            `Dự báo cho: ${data.target_date} (${data.target_weekday})`;
        document.getElementById("forecast-method").innerText = data.method;
        document.getElementById("forecast-total-val").innerText = 
            data.predicted_total !== null ? `${data.predicted_total} suất` : "Nghỉ lễ/cuối tuần";

        document.getElementById("best-model-name").innerText = data.model;
        document.getElementById("best-model-acc").innerText = 
            data.target_accuracy !== null ? `${data.target_accuracy}%` : "Chưa có";
        document.getElementById("best-model-range").innerText = 
            `${data.overall_min_servings ?? 0} - ${data.overall_max_servings ?? 0} suất`;

        // Render phân rã danh mục món ăn (Việt, Âu, Chay)
        const catContainer = document.getElementById("category-cards");
        catContainer.innerHTML = "";
        (data.categories || []).forEach(cat => {
            const card = document.createElement("div");
            card.className = "metric-card";
            const acc = cat.metrics && cat.metrics.accuracy !== null ? `${cat.metrics.accuracy}%` : "N/A";
            card.innerHTML = `
                <span class="metric-label">Món ${cat.category}</span>
                <span class="metric-value">${cat.predicted_servings !== null ? cat.predicted_servings : 0} suất</span>
                <span class="metric-sub">Độ chính xác lịch sử: ${acc}</span>
            `;
            catContainer.appendChild(card);
        });

        // Render Leaderboard
        const tableBody = document.querySelector("#leaderboard-table tbody");
        tableBody.innerHTML = "";
        (data.model_leaderboard || []).forEach(item => {
            const tr = document.createElement("tr");
            if (item.model === data.model) tr.classList.add("highlight-row");
            tr.innerHTML = `
                <td><strong>${item.model}</strong> ${item.model === data.model ? "⭐ (Được chọn)" : ""}</td>
                <td>${item.mape}%</td>
                <td>${item.accuracy}%</td>
                <td>${item.samples}</td>
            `;
            tableBody.appendChild(tr);
        });
    } catch (e) {
        console.error("Lỗi khi chạy dự báo:", e);
    }
}

// 6. Tải và phân trang bảng dữ liệu lịch sử
async function fetchRecordsData() {
    const search = document.getElementById("search-input").value;
    const department = document.getElementById("department-select").value;
    const status = document.getElementById("status-select").value;

    const params = new URLSearchParams({
        page: currentPage,
        page_size: pageSize,
        search: search,
        department: department,
        status: status
    });

    try {
        const res = await fetch(`/api/records?${params.toString()}`);
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.getElementById("records-body");
        tbody.innerHTML = "";

        if (!data.items || data.items.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="text-center">Không tìm thấy bản ghi phù hợp.</td></tr>`;
        } else {
            data.items.forEach(row => {
                const tr = document.createElement("tr");
                const isValid = row["Trạng thái"] === "Hợp lệ";
                tr.innerHTML = `
                    <td>${row["Thời gian ăn"] || "-"}</td>
                    <td>${row["Món ăn"] || "-"}</td>
                    <td>${row["Phòng ban"] || "-"}</td>
                    <td><span class="badge ${isValid ? 'badge-success' : 'badge-danger'}">${row["Trạng thái"] || "-"}</span></td>
                    <td>${Number(row["Thành tiền"] || 0).toLocaleString("vi-VN")} đ</td>
                `;
                tbody.appendChild(tr);
            });
        }

        // Cập nhật phân trang
        const totalPages = Math.ceil(data.total / pageSize) || 1;
        document.getElementById("pagination-info").innerText = 
            `Trang ${data.page} / ${totalPages} (Tổng ${Number(data.total).toLocaleString("vi-VN")} dòng)`;

        document.getElementById("btn-prev").disabled = data.page <= 1;
        document.getElementById("btn-next").disabled = data.page >= totalPages;
    } catch (e) {
        console.error("Lỗi khi nạp danh sách bản ghi:", e);
    }
}