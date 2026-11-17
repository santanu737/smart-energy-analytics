const API = {
    getToken() {
        return localStorage.getItem('access_token');
    },

    async request(endpoint) {
        const token = this.getToken();
        if (!token) {
            window.location.href = '/login.html';
            return;
        }

        const res = await fetch(endpoint, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (res.status === 401) {
            localStorage.removeItem('access_token');
            window.location.href = '/login.html';
            return;
        }

        if (!res.ok) throw new Error('API request failed');
        return res.json();
    },

    getSummary() { return this.request('/api/campus/summary'); },
    getHistory(resolution = 'daily') { return this.request(`/api/campus/history?resolution=${resolution}`); },
    getUsageTypes() { return this.request('/api/usage-types'); },
    getRanking() { return this.request('/api/ranking?metric=eui&limit=5'); },
    getAnomalies() { return this.request('/api/anomalies?limit=5'); },
    getCostCarbon() { return this.request('/api/cost-carbon'); },
    getForecast(horizon = 24) { return this.request(`/api/forecast?horizon=${horizon}&points=24`); },
    getBuildings() { return this.request('/api/buildings'); }
};
