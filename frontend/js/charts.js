let campusChartInstance = null;
let usageDoughnutInstance = null;

const Charts = {
    renderCampusChart(data) {
        const ctx = document.getElementById('campusChart').getContext('2d');
        const labels = data.map(d => d.timestamp.split('T')[0]);
        const values = data.map(d => d.kwh);

        if (campusChartInstance) campusChartInstance.destroy();

        campusChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Campus Daily Energy (kWh)',
                    data: values,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    },

    renderUsageDoughnut(data) {
        const ctx = document.getElementById('usageDoughnut').getContext('2d');
        const labels = Object.keys(data);
        const values = Object.values(data);

        if (usageDoughnutInstance) usageDoughnutInstance.destroy();

        usageDoughnutInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#1e40af']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
};
