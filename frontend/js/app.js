document.addEventListener('DOMContentLoaded', async () => {
    if (!API.getToken()) {
        window.location.href = '/login.html';
        return;
    }

    // Logout
    document.getElementById('logout-btn').addEventListener('click', () => {
        localStorage.removeItem('access_token');
        window.location.href = '/login.html';
    });

    // Theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    const currentTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);

    themeToggle.addEventListener('click', () => {
        const theme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    });

    const sections = [...document.querySelectorAll('.dashboard-section')];
    const navItems = [...document.querySelectorAll('.sidebar-menu li')];
    const titles = {
        overview: 'Campus Overview',
        forecast: 'Load Forecasting',
        buildings: 'Building Performance',
        anomalies: 'Anomaly Detection'
    };

    async function showSection(target) {
        navItems.forEach(item => item.classList.toggle('active', item.dataset.target === target));
        sections.forEach(section => { section.hidden = section.id !== target; });
        const selectedSection = document.getElementById(target);
        document.getElementById('page-title').textContent = titles[target];
        selectedSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        if (target === 'forecast') await loadForecast();
        if (target === 'buildings') await loadBuildings();
        if (target === 'anomalies') await loadAllAnomalies();
    }

    navItems.forEach(item => {
        item.addEventListener('click', () => showSection(item.dataset.target));
        item.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                showSection(item.dataset.target);
            }
        });
    });

    async function loadForecast() {
        const body = document.querySelector('#forecast-table tbody');
        body.innerHTML = '<tr><td colspan="3">Loading...</td></tr>';
        try {
            const result = await API.getForecast();
            body.innerHTML = result.points.map(point => `
                <tr>
                    <td>${point.target_timestamp}</td>
                    <td>${Number(point.forecast).toLocaleString(undefined, {maximumFractionDigits: 1})}</td>
                    <td>${point.actual == null ? '--' : Number(point.actual).toLocaleString(undefined, {maximumFractionDigits: 1})}</td>
                </tr>`).join('');
        } catch (error) {
            body.innerHTML = `<tr><td colspan="3">${error.message}</td></tr>`;
        }
    }

    async function loadBuildings() {
        const body = document.querySelector('#buildings-table tbody');
        body.innerHTML = '<tr><td colspan="5">Loading...</td></tr>';
        try {
            const buildings = await API.getBuildings();
            body.innerHTML = buildings.map(building => `
                <tr>
                    <td>${building.building_id}</td>
                    <td>${building.primaryspaceusage || '--'}</td>
                    <td>${Number(building.total_kwh).toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                    <td>${building.eui_annual == null ? '--' : Number(building.eui_annual).toFixed(1)}</td>
                    <td>${building.baseload_ratio == null ? '--' : Number(building.baseload_ratio).toFixed(2)}</td>
                </tr>`).join('');
        } catch (error) {
            body.innerHTML = `<tr><td colspan="5">${error.message}</td></tr>`;
        }
    }

    async function loadAllAnomalies() {
        const body = document.querySelector('#all-anomalies-table tbody');
        body.innerHTML = '<tr><td colspan="5">Loading...</td></tr>';
        try {
            const anomalies = await API.getAnomalies();
            body.innerHTML = anomalies.map(anomaly => `
                <tr class="anomaly-row">
                    <td>${anomaly.building_id}</td>
                    <td>${anomaly.timestamp}</td>
                    <td>${Number(anomaly.kwh).toFixed(1)}</td>
                    <td>${Number(anomaly.anomaly_score).toFixed(2)}</td>
                    <td>${anomaly.source || '--'}</td>
                </tr>`).join('');
        } catch (error) {
            body.innerHTML = `<tr><td colspan="5">${error.message}</td></tr>`;
        }
    }

    document.getElementById('refresh-forecast').addEventListener('click', loadForecast);

    try {
        // Load KPI Summary
        const summary = await API.getSummary();
        document.getElementById('kpi-total-energy').textContent = summary.total_kwh.toLocaleString(undefined, {maximumFractionDigits: 0}) + ' kWh';
        document.getElementById('kpi-avg-load').textContent = summary.avg_load_kw.toLocaleString(undefined, {maximumFractionDigits: 1}) + ' kW';
        document.getElementById('kpi-peak-load').textContent = summary.peak_load_kw.toLocaleString(undefined, {maximumFractionDigits: 1}) + ' kW';
        document.getElementById('kpi-anomalies').textContent = summary.anomaly_count;

        // Load Cost & Carbon
        const costCarbon = await API.getCostCarbon();
        document.getElementById('kpi-cost').textContent = '$' + costCarbon.estimated_cost.toLocaleString(undefined, {maximumFractionDigits: 0});

        // Load Charts
        const history = await API.getHistory('daily');
        Charts.renderCampusChart(history);

        const usageTypes = await API.getUsageTypes();
        Charts.renderUsageDoughnut(usageTypes);

        // Load Ranking Table
        const ranking = await API.getRanking();
        const rankingBody = document.querySelector('#ranking-table tbody');
        rankingBody.innerHTML = ranking.map(r => `
            <tr>
                <td>${r.building_id}</td>
                <td>${r.primaryspaceusage}</td>
                <td>${r.eui_annual.toFixed(1)}</td>
                <td>${r.baseload_ratio.toFixed(2)}</td>
            </tr>
        `).join('');

        // Load Anomaly Table
        const anomalies = await API.getAnomalies();
        const anomalyBody = document.querySelector('#anomaly-table tbody');
        anomalyBody.innerHTML = anomalies.map(a => `
            <tr class="anomaly-row">
                <td>${a.building_id}</td>
                <td>${a.timestamp}</td>
                <td>${a.kwh.toFixed(1)}</td>
                <td>${a.anomaly_score.toFixed(2)}</td>
            </tr>
        `).join('');

    } catch (err) {
        console.error('Failed to load dashboard data:', err);
    }
});
