const $ = (id) => document.getElementById(id);
const money = new Intl.NumberFormat("vi-VN", { style: "currency", currency: "VND" });

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error((await response.json()).detail || "API error");
  return response.json();
}

async function loadSummary() {
  const data = await api("/api/summary");
  $("total").textContent = data.rows.toLocaleString("vi-VN");
  $("valid").textContent = data.valid_rows.toLocaleString("vi-VN");
  $("invalid").textContent = data.invalid_rows.toLocaleString("vi-VN");
  $("amount").textContent = money.format(data.total_amount);
}

function metric(value, suffix = "") {
  return value == null ? "Chưa đủ dữ liệu" : `${value.toLocaleString("vi-VN")}${suffix}`;
}

async function loadForecast() {
  const data = await api("/api/forecast");
  const target = new Date(`${data.target_date}T00:00:00`);
  const label = target.toLocaleDateString("vi-VN", { weekday: "long", day: "numeric", month: "numeric" });
  $("forecast-total").textContent = data.predicted_total == null ? "Chưa đủ dữ liệu" : data.predicted_total.toLocaleString("vi-VN");
  $("forecast-detail").textContent = `${label} · ${data.method}`;
  $("forecast-confidence").textContent = data.metrics.accuracy == null ? "Chưa xác định" : `Accuracy ${data.metrics.accuracy}%`;
  $("model-note").textContent = `Mô hình được chọn: ${data.model}. Mục tiêu Accuracy ~75%: ${data.target_accuracy == null ? "chưa đủ dữ liệu" : data.target_accuracy >= 75 ? "đạt trên backtest" : "chưa đạt trên backtest"}.`;
  $("forecast-samples").textContent = data.overall_min_servings == null
    ? `Dựa trên ${data.history_days} ngày lịch sử`
    : `Khoảng tổng: ${data.overall_min_servings.toLocaleString("vi-VN")} – ${data.overall_max_servings.toLocaleString("vi-VN")} · ${data.history_days} ngày`;
  $("forecast-table").innerHTML = data.categories.map((item) => `
    <tr><td><strong>${item.category}</strong></td><td>${metric(item.predicted_servings, " suất")}</td>
    <td>${item.min_servings == null ? "—" : `${item.min_servings} – ${item.max_servings}`}</td>
    <td>${item.sample_days}</td><td><span class="badge ${item.status === "ok" ? "valid" : "invalid"}">${item.status === "ok" ? "Có dữ liệu" : "Thiếu dữ liệu"}</span></td></tr>`).join("");
  const metrics = [["MAE", data.metrics.mae, "Sai lệch tuyệt đối trung bình (suất)"], ["RMSE", data.metrics.rmse, "Phạt sai lệch lớn"], ["MAPE", data.metrics.mape, "Sai số phần trăm trung bình"], ["Accuracy", data.metrics.accuracy, "1 - MAPE, chỉ số tham khảo"], ["Độ bao phủ khoảng", data.interval_metrics.coverage, "Tỷ lệ ngày thực tế nằm trong khoảng dự báo"], ["Độ rộng khoảng", data.interval_metrics.average_width, "Khoảng dự báo rộng trung bình (suất)"]];
  $("metrics-table").innerHTML = metrics.map(([name, value, meaning]) => `<tr><td><strong>${name}</strong></td><td>${metric(value, name === "MAPE" || name === "Accuracy" || name === "Độ bao phủ khoảng" ? "%" : "")}</td><td>${meaning}</td></tr>`).join("");
  $("models-table").innerHTML = data.model_leaderboard.length
    ? data.model_leaderboard.map((item) => `<tr><td>${item.model}</td><td>${item.mape}%</td><td>${item.accuracy}%</td><td>${item.samples}</td></tr>`).join("")
    : '<tr><td colspan="4">Chưa đủ dữ liệu để so sánh mô hình.</td></tr>';
  $("history-table").innerHTML = `<tr><td>Tất cả nhóm món</td><td>${metric(data.metrics.mae)}</td><td>${metric(data.metrics.rmse)}</td><td>${metric(data.metrics.mape, "%")}</td><td>${metric(data.metrics.accuracy, "%")}</td></tr>`;
  $("history-summary").textContent = `${data.history_days.toLocaleString("vi-VN")} ngày lịch sử tổng · ${data.categorized_rows.toLocaleString("vi-VN")} dòng đã phân loại món`;
  $("data-warning-text").textContent = data.uncategorized_valid_rows
    ? `${data.uncategorized_valid_rows.toLocaleString("vi-VN")} dòng hợp lệ chưa có nhãn món Việt/Âu/Chay; dự báo có thể chưa đầy đủ.`
    : "Tất cả dòng hợp lệ đã được phân loại.";
}

async function init() {
  try {
    await Promise.all([loadSummary(), loadForecast()]);
    $("health").textContent = "● Hệ thống hoạt động";
    $("health").classList.add("online");
  } catch (error) {
    $("health").textContent = "Không thể kết nối API";
    $("forecast-total").textContent = error.message;
  }
}

init();
