import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import webbrowser
import tempfile
from pathlib import Path

import support_files.updated_config as config

# Utility: Safe date conversion and dashboard file export
def _safe_date_convert(date_val):
    """Convert dates safely into ISO format for dashboard export."""
    try:
        if isinstance(date_val, str):
            ts = pd.to_datetime(date_val, errors="coerce")
        else:
            ts = pd.to_datetime(date_val, errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(date_val)

def export_dashboard_data(all_stock_data, output_dir="output_data"):
    """Export stock data into JSON for dashboard with proper Plotly format.
    Note: Caller should pass the final folder (e.g., 'dashboard_exports').
    """
    os.makedirs(output_dir, exist_ok=True)
    export_path = os.path.join(output_dir, "trading_data.json")

    all_charts = {}

    for stock, stock_df in all_stock_data.items():
        df = stock_df.copy()
        # Ensure ISO date strings and drop bad dates
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["Date"])  # drop rows with invalid dates

        chart_data = {
            "data": [
                {
                    "type": "candlestick",
                    "x": df["Date"].tolist(),
                    "open": df.get("Open", pd.Series(dtype=float)).tolist(),
                    "high": df.get("High", pd.Series(dtype=float)).tolist(),
                    "low": df.get("Low", pd.Series(dtype=float)).tolist(),
                    "close": df.get("Close", pd.Series(dtype=float)).tolist(),
                    "name": stock,
                }
            ],
            "layout": {
                "title": f"{stock} Candlestick Chart",
                "xaxis": {
                    "title": "Date",
                    "rangeslider": {"visible": False},
                    "anchor": "y"
                },
                "yaxis": {
                    "title": "Price",
                    "anchor": "x"
                },
            },
        }

        all_charts[stock] = chart_data

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(all_charts, f, indent=2)

    # Single output only (avoid accidental nested output_data paths)
    print(f"[OK] Dashboard JSON exported → {export_path}")

def export_dashboard_files(html_content, trading_data):
    """Export dashboard HTML and JSON to dashboard_exports with safe date normalization."""
    export_dir = os.path.join(os.getcwd(), "dashboard_exports")
    os.makedirs(export_dir, exist_ok=True)

    html_path = os.path.join(export_dir, "trading_dashboard.html")
    json_path = os.path.join(export_dir, "trading_data.json")

    # Normalize dates before export (handle a simple mapping of stock->list[dict])
    try:
        for stock, entries in (trading_data.items() if isinstance(trading_data, dict) else []):
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and "Date" in entry:
                        entry["Date"] = _safe_date_convert(entry["Date"])
    except Exception:
        # Best-effort normalization; proceed even if structure differs
        pass

    # Save HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(trading_data, f, indent=4)

    print(f"[OK] Dashboard exported to {export_dir}")
    print(f"   - HTML Dashboard: {html_path}")
    print(f"   - Trading Data: {json_path}")

class TradingDashboard:
    def __init__(self, backtested_scrips_df, backtested_transactions_df, strategy_name=None):
        """
        Initialize the Trading Dashboard with backtested data.
        
        Args:
            backtested_scrips_df: DataFrame containing OHLC data with buy/sell signals
            backtested_transactions_df: DataFrame containing transaction history
        """
        self.backtested_scrips_df = backtested_scrips_df
        self.backtested_transactions_df = backtested_transactions_df
        self.strategy_name = strategy_name
        self.dashboard_data = {}
        self.prepare_dashboard_data()
    
    def prepare_dashboard_data(self):
        """
        Convert the backtested data into format suitable for the dashboard.
        """
        print("📄 Preparing data for visualization dashboard...")
        
        stock_data = {}
        transaction_data = {}
        
        if self.backtested_scrips_df.empty:
            print("⚠️ No backtested scrip data available")
            return
        
        # Group data by stock
        unique_stocks = self.backtested_scrips_df['Stock'].unique()
        
        for stock in unique_stocks:
            stock_df = self.backtested_scrips_df[self.backtested_scrips_df['Stock'] == stock].copy()
            
            # Sort by date
            if 'Date' in stock_df.columns:
                # Parse Date in ISO (YYYY-MM-DD) only
                stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce', dayfirst=False)

                # Drop rows where date is missing
                stock_df = stock_df.dropna(subset=['Date'])

                # Ensure sorted by Date
                stock_df = stock_df.sort_values(by='Date').reset_index(drop=True)

                # Debug prints removed for cleaner logs

                # Ensure Date column is fully cleaned and ISO formatted
                stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
                stock_df['Date'] = stock_df['Date'].ffill()  # fill NaNs forward

                # Export dates as ISO strings for JSON and Excel compatibility
                stock_df['Date'] = stock_df['Date'].dt.strftime('%Y-%m-%d')

                # Force conversion to string type to avoid JS parse issues
                stock_df['Date'] = stock_df['Date'].astype(str)

                # Debug prints removed for cleaner logs
            else:
                # Use index as date if Date column is not available
                stock_df = stock_df.sort_index()
                stock_df['Date'] = stock_df.index
            
            # Prepare OHLC data with signals
            chart_data = []
            for idx, row in stock_df.iterrows():
                try:
                    # Handle different date formats
                    if pd.isna(row['Date']):
                        continue
                        
                    date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], pd.Timestamp) else str(row['Date'])
                    
                    chart_data.append({
                        'date': date_str,
                        'open': float(row.get('Open', row.get('open', 0))),
                        'high': float(row.get('High', row.get('high', 0))),
                        'low': float(row.get('Low', row.get('low', 0))),
                        'close': float(row.get('Close', row.get('close', 0))),
                        'volume': int(row.get('Volume', row.get('volume', 1000000))),
                        'buySignal': bool(row.get('Buy', False)) if pd.notna(row.get('Buy')) else False,
                        'sellSignal': bool(row.get('Sell', False)) if pd.notna(row.get('Sell')) else False
                    })
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Error processing row for {stock}: {e}")
                    continue
            
            stock_data[stock] = chart_data
            
            # Prepare transaction data
            if not self.backtested_transactions_df.empty:
                stock_transactions = self.backtested_transactions_df[
                    self.backtested_transactions_df['Stock'] == stock
                ].copy()
                
                transactions = []
                for idx, row in stock_transactions.iterrows():
                    try:
                        # Handle date formatting
                        trans_date = row.get('Date')
                        if pd.isna(trans_date):
                            continue
                            
                        if isinstance(trans_date, pd.Timestamp):
                            date_str = trans_date.strftime('%Y-%m-%d')
                        else:
                            date_str = pd.to_datetime(str(trans_date), errors='coerce').strftime('%Y-%m-%d')
                        
                        action = str(row.get('Event', row.get('Action', 'UNKNOWN'))).upper()
                        price = float(row.get('Price', 0))
                        quantity = int(row.get('Shares', row.get('Quantity', 0)))
                        
                        # Calculate amount based on action
                        if action == 'BUY':
                            amount = float(row.get('Cost', row.get('Revenue', price * quantity)))
                        else:
                            amount = float(row.get('Revenue', row.get('Cost', price * quantity)))
                        
                        # Calculate P&L and holding period
                        pnl = float(row.get('Profit_%', 0))
                        if action == 'SELL' and 'Profit_%' not in row:
                            # If no profit percentage, calculate from amount
                            pnl = amount - (price * quantity)
                        
                        holding_days = int(row.get('Holding_Period', 0))
                        
                        transactions.append({
                            'date': date_str,
                            'action': action,
                            'price': round(price, 2),
                            'quantity': quantity,
                            'amount': round(amount, 2),
                            'pnl': round(pnl, 2),
                            'holdingDays': holding_days
                        })
                        
                    except (ValueError, TypeError, KeyError) as e:
                        print(f"⚠️ Error processing transaction for {stock}: {e}")
                        continue
                
                transaction_data[stock] = transactions
            else:
                transaction_data[stock] = []
        
        # Persist both granular and combined structures
        self.stock_data = stock_data
        self.transactions_data = transaction_data
        self.dashboard_data = {
            'stockData': stock_data,
            'transactionData': transaction_data
        }
        
        print(f"✅ Dashboard data prepared for {len(unique_stocks)} stocks")
    
    def save_data_for_dashboard(self, filename='dashboard_data.json'):
        """
        Save the prepared data to a JSON file for the dashboard.
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.dashboard_data, f, indent=2, default=str)
            print(f"✅ Dashboard data saved to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving dashboard data: {e}")
            return None
    
    def create_dashboard_html(self, include_data=True):
        """
        Create a complete HTML dashboard with embedded data.
        """
        # Read the base HTML template (from the artifact)
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Portfolio Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: #333;
        }
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px; text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            position: sticky; top: 0; z-index: 100;
        }
        .header h1 {
            color: #2c3e50; font-size: 2.5em; font-weight: 700;
            margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .header p { color: #7f8c8d; font-size: 1.2em; font-weight: 300; }
        .container { max-width: 1400px; margin: 0 auto; padding: 30px 20px; }
        .controls {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px); border-radius: 20px;
            padding: 25px; margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .controls h3 { margin-bottom: 20px; color: #2c3e50; font-size: 1.4em; }
        .control-group { display: flex; flex-wrap: wrap; gap: 20px; align-items: center; }
        .control-item { display: flex; flex-direction: column; gap: 8px; min-width: 200px; }
        label { font-weight: 600; color: #34495e; font-size: 0.9em; }
        select, input, button {
            padding: 12px 15px; border: 2px solid #e0e6ed; border-radius: 12px;
            font-size: 14px; transition: all 0.3s ease; background: white;
        }
        select:focus, input:focus {
            outline: none; border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; cursor: pointer; font-weight: 600;
            text-transform: uppercase; letter-spacing: 1px; min-width: 150px;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); }
        .chart-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px); border-radius: 20px;
            padding: 25px; margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .chart-title { font-size: 1.8em; font-weight: 700; color: #2c3e50; margin-bottom: 20px; text-align: center; }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px); border-radius: 15px;
            padding: 20px; text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-value { font-size: 2.2em; font-weight: 700; margin-bottom: 5px; }
        .stat-label { color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .neutral { color: #3498db; }
        .loading { display: none; text-align: center; padding: 50px; font-size: 1.2em; color: #7f8c8d; }
        .spinner {
            border: 4px solid #f3f3f3; border-top: 4px solid #667eea;
            border-radius: 50%; width: 50px; height: 50px;
            animation: spin 1s linear infinite; margin: 20px auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .transaction-details {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px); border-radius: 15px;
            padding: 20px; margin-top: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .transaction-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        .transaction-table th, .transaction-table td {
            padding: 12px; text-align: left; border-bottom: 1px solid #e0e6ed;
        }
        .transaction-table th { background: #f8f9fa; font-weight: 600; color: #2c3e50; }
        .buy-signal { background-color: rgba(39, 174, 96, 0.1); }
        .sell-signal { background-color: rgba(231, 76, 60, 0.1); }
        .real-data-banner {
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            color: white; text-align: center; padding: 10px;
            font-weight: bold; margin-bottom: 20px;
            border-radius: 10px; box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
        }
        @media (max-width: 768px) {
            .control-group { flex-direction: column; }
            .control-item { min-width: 100%; }
            .header h1 { font-size: 2em; }
            .container { padding: 20px 10px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Trading Portfolio Dashboard</h1>
        <p>Interactive Candlestick Charts with Buy/Sell Signals</p>
    </div>

    <div class="container">
        <div class="real-data-banner">
            🎯 <strong>LIVE TRADING DATA</strong> - Generated from your actual backtesting results
        </div>
        
        <div class="controls">
            <h3>📈 Chart Controls</h3>
            <div class="control-group">
                <div class="control-item">
                    <label for="stockSelect">Select Stock:</label>
                    <select id="stockSelect">
                        <option value="">Choose a stock...</option>
                    </select>
                </div>
                <div class="control-item">
                    <label for="dateRange">Date Range (days):</label>
                    <input type="number" id="dateRange" min="30" max="365" value="90" placeholder="90">
                </div>
                <div class="control-item">
                    <label for="showVolume">Show Volume:</label>
                    <select id="showVolume">
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                    </select>
                </div>
                <div class="control-item">
                    <button onclick="generateChart()">📊 Generate Chart</button>
                </div>
            </div>
        </div>

        <div id="portfolioStats" class="stats-grid" style="display: none;">
            <div class="stat-card">
                <div class="stat-value neutral" id="totalTrades">0</div>
                <div class="stat-label">Total Trades</div>
            </div>
            <div class="stat-card">
                <div class="stat-value positive" id="winRate">0%</div>
                <div class="stat-label">Win Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value neutral" id="avgHolding">0</div>
                <div class="stat-label">Avg Holding (Days)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value positive" id="totalPnL">₹0</div>
                <div class="stat-label">Total P&L</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title" id="chartTitle">Select a stock to view chart</div>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Generating chart...</p>
            </div>
            <div id="candlestickChart"></div>
        </div>

        <div id="transactionDetails" class="transaction-details" style="display: none;">
            <h3>💼 Transaction History</h3>
            <table class="transaction-table" id="transactionTable">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Action</th>
                        <th>Price</th>
                        <th>Quantity</th>
                        <th>Amount</th>
                        <th>P&L</th>
                        <th>Holding Days</th>
                    </tr>
                </thead>
                <tbody id="transactionTableBody">
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Embedded real trading data
        let stockData = STOCK_DATA_PLACEHOLDER;
        let transactionData = TRANSACTION_DATA_PLACEHOLDER;
        let currentStock = '';

        function populateStockSelect() {
            const select = document.getElementById('stockSelect');
            select.innerHTML = '<option value="">Choose a stock...</option>';
            
            Object.keys(stockData).forEach(stock => {
                const option = document.createElement('option');
                option.value = stock;
                option.textContent = stock;
                select.appendChild(option);
            });
        }

        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }

        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }

        function generateChart() {
            const selectedStock = document.getElementById('stockSelect').value;
            const dateRange = parseInt(document.getElementById('dateRange').value) || 90;
            const showVolume = document.getElementById('showVolume').value === 'true';
            
            if (!selectedStock) {
                alert('Please select a stock first!');
                return;
            }
            
            if (!stockData[selectedStock]) {
                alert('No data available for selected stock.');
                return;
            }
            
            showLoading();
            currentStock = selectedStock;
            
            setTimeout(() => {
                const data = stockData[selectedStock];
                const transactions = transactionData[selectedStock] || [];
                
                // Filter data by date range
                const cutoffDate = new Date();
                cutoffDate.setDate(cutoffDate.getDate() - dateRange);
                
                const filteredData = data.filter(d => new Date(d.date) >= cutoffDate);
                
                createCandlestickChart(filteredData, transactions, selectedStock, showVolume);
                updatePortfolioStats(transactions);
                updateTransactionTable(transactions);
                
                document.getElementById('chartTitle').textContent = `${selectedStock} - Live Trading Results`;
                document.getElementById('portfolioStats').style.display = 'grid';
                document.getElementById('transactionDetails').style.display = 'block';
                
                hideLoading();
            }, 500);
        }

        function createCandlestickChart(data, transactions, stockName, showVolume) {
            const dates = data.map(d => d.date);
            const opens = data.map(d => d.open);
            const highs = data.map(d => d.high);
            const lows = data.map(d => d.low);
            const closes = data.map(d => d.close);
            const volumes = data.map(d => d.volume);
            
            // Buy signals
            const buyDates = [];
            const buyPrices = [];
            const buyTexts = [];
            
            // Sell signals
            const sellDates = [];
            const sellPrices = [];
            const sellTexts = [];
            
            data.forEach(d => {
                if (d.buySignal) {
                    buyDates.push(d.date);
                    buyPrices.push(d.low * 0.98);
                    const transaction = transactions.find(t => t.date === d.date && t.action === 'BUY');
                    buyTexts.push(transaction ? 
                        `BUY: ${transaction.quantity} shares<br>Price: ₹${transaction.price}<br>Amount: ₹${transaction.amount.toLocaleString()}` :
                        `BUY Signal<br>Price: ₹${d.close.toFixed(2)}`
                    );
                }
                if (d.sellSignal) {
                    sellDates.push(d.date);
                    sellPrices.push(d.high * 1.02);
                    const transaction = transactions.find(t => t.date === d.date && t.action === 'SELL');
                    sellTexts.push(transaction ? 
                        `SELL: ${transaction.quantity} shares<br>Price: ₹${transaction.price}<br>Amount: ₹${transaction.amount.toLocaleString()}<br>P&L: ₹${transaction.pnl.toLocaleString()}<br>Holding: ${transaction.holdingDays} days` :
                        `SELL Signal<br>Price: ₹${d.close.toFixed(2)}`
                    );
                }
            });

            const traces = [
                {
                    type: 'candlestick',
                    x: dates,
                    open: opens,
                    high: highs,
                    low: lows,
                    close: closes,
                    name: stockName,
                    increasing: {fillcolor: '#26a69a', line: {color: '#26a69a'}},
                    decreasing: {fillcolor: '#ef5350', line: {color: '#ef5350'}},
                    yaxis: 'y'
                },
                {
                    type: 'scatter',
                    mode: 'markers',
                    x: buyDates,
                    y: buyPrices,
                    marker: {
                        symbol: 'triangle-up',
                        size: 15,
                        color: '#27ae60',
                        line: {color: '#ffffff', width: 2}
                    },
                    text: buyTexts,
                    hovertemplate: '%{text}<extra></extra>',
                    name: 'Buy Signals',
                    yaxis: 'y'
                },
                {
                    type: 'scatter',
                    mode: 'markers',
                    x: sellDates,
                    y: sellPrices,
                    marker: {
                        symbol: 'triangle-down',
                        size: 15,
                        color: '#e74c3c',
                        line: {color: '#ffffff', width: 2}
                    },
                    text: sellTexts,
                    hovertemplate: '%{text}<extra></extra>',
                    name: 'Sell Signals',
                    yaxis: 'y'
                }
            ];

            if (showVolume && volumes.some(v => v > 0)) {
                traces.push({
                    type: 'bar',
                    x: dates,
                    y: volumes,
                    name: 'Volume',
                    marker: {color: 'rgba(70,130,180,0.5)'},
                    yaxis: 'y2'
                });
            }

            const layout = {
                title: {
                    text: `${stockName} - Live Trading Results`,
                    font: {size: 20, color: '#2c3e50'}
                },
                xaxis: {
                    title: 'Date',
                    type: 'date',
                    gridcolor: '#e0e6ed',
                    tickfont: {color: '#34495e'}
                },
                yaxis: {
                    title: 'Price (₹)',
                    side: 'left',
                    gridcolor: '#e0e6ed',
                    tickfont: {color: '#34495e'}
                },
                yaxis2: showVolume && volumes.some(v => v > 0) ? {
                    title: 'Volume',
                    side: 'right',
                    overlaying: 'y',
                    showgrid: false,
                    tickfont: {color: '#34495e'}
                } : undefined,
                plot_bgcolor: 'rgba(255,255,255,0.8)',
                paper_bgcolor: 'rgba(255,255,255,0.95)',
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1,
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#e0e6ed',
                    borderwidth: 1
                },
                hovermode: 'x unified',
                dragmode: 'zoom',
                responsive: true
            };

            const config = {
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            };

            Plotly.newPlot('candlestickChart', traces, layout, config);
        }

        function updatePortfolioStats(transactions) {
            const sellTrades = transactions.filter(t => t.action === 'SELL');
            const totalTrades = sellTrades.length;
            const profitableTrades = sellTrades.filter(t => t.pnl > 0).length;
            const winRate = totalTrades > 0 ? ((profitableTrades / totalTrades) * 100).toFixed(1) : 0;
            
            const avgHolding = totalTrades > 0 ? 
                (sellTrades.reduce((sum, t) => sum + t.holdingDays, 0) / totalTrades).toFixed(0) : 0;
            
            const totalPnL = sellTrades.reduce((sum, t) => sum + t.pnl, 0);

            document.getElementById('totalTrades').textContent = totalTrades;
            document.getElementById('winRate').textContent = `${winRate}%`;
            document.getElementById('avgHolding').textContent = avgHolding;
            document.getElementById('totalPnL').textContent = `₹${totalPnL.toLocaleString()}`;
            
            // Update colors based on performance
            document.getElementById('winRate').className = `stat-value ${winRate >= 50 ? 'positive' : 'negative'}`;
            document.getElementById('totalPnL').className = `stat-value ${totalPnL >= 0 ? 'positive' : 'negative'}`;
        }

        function updateTransactionTable(transactions) {
            const tbody = document.getElementById('transactionTableBody');
            tbody.innerHTML = '';
            
            transactions.slice(-20).reverse().forEach(transaction => {
                const row = document.createElement('tr');
                row.className = transaction.action === 'BUY' ? 'buy-signal' : 'sell-signal';
                
                row.innerHTML = `
                    <td>${new Date(transaction.date).toLocaleDateString()}</td>
                    <td><strong>${transaction.action}</strong></td>
                    <td>₹${transaction.price.toLocaleString()}</td>
                    <td>${transaction.quantity.toLocaleString()}</td>
                    <td>₹${transaction.amount.toLocaleString()}</td>
                    <td class="${transaction.pnl >= 0 ? 'positive' : 'negative'}">
                        ${transaction.pnl !== 0 ? `₹${transaction.pnl.toLocaleString()}` : '-'}
                    </td>
                    <td>${transaction.holdingDays || '-'}</td>
                `;
                
                tbody.appendChild(row);
            });
        }

        // Initialize the dashboard
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Live Trading Dashboard Loaded');
            populateStockSelect();
            
            // Automatically select and show first stock if available
            const stocks = Object.keys(stockData);
            if (stocks.length > 0) {
                document.getElementById('stockSelect').value = stocks[0];
                generateChart();
            }
            
            // Add event listeners
            document.getElementById('stockSelect').addEventListener('change', function() {
                if (this.value) {
                    generateChart();
                }
            });
        });
    </script>
</body>
</html>"""
        
        if include_data:
            # Embed the actual data into the HTML
            stock_data_json = json.dumps(self.dashboard_data['stockData'])
            transaction_data_json = json.dumps(self.dashboard_data['transactionData'])
            
            html_template = html_template.replace('STOCK_DATA_PLACEHOLDER', stock_data_json)
            html_template = html_template.replace('TRANSACTION_DATA_PLACEHOLDER', transaction_data_json)
        
        # If a strategy name was provided, insert a banner under the LIVE DATA banner
        if getattr(self, 'strategy_name', None):
            strategy_banner = f"""
        <div class=\"real-data-banner\" style=\"background: linear-gradient(45deg, #3498db, #2980b9);\">
            🧭 <strong>Optimiser Strategy</strong>: {self.strategy_name}
        </div>
"""
            marker = "</div>\n        \n        <div class=\"controls\">"
            html_template = html_template.replace("</div>\n        \n        <div class=\"controls\">", f"</div>\n{strategy_banner}        \n        <div class=\\\"controls\\\">", 1)
        
        return html_template
    
    def launch_dashboard(self, export_dir='output_data/dashboard_exports', filename='trading_dashboard.html', auto_open=True):
        """
        Create and launch the dashboard in the default web browser.
        """
        print("🚀 Launching Trading Dashboard...")
        
        try:
            # Create HTML with embedded data
            html_content = self.create_dashboard_html(include_data=True)
            
            # Save to file under the provided export_dir
            try:
                pkg_dir = Path(__file__).resolve().parent
                os.chdir(pkg_dir)
            except Exception:
                pass
            out_dir = Path(export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            file_path = out_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            file_path = os.path.abspath(str(file_path))
            print(f"✅ Dashboard created: {file_path}")
            
            if auto_open:
                webbrowser.open(f'file://{file_path}')
                print("🌐 Dashboard opened in your default browser")
            
            return file_path
            
        except Exception as e:
            print(f"❌ Error launching dashboard: {e}")
            return None

    def export_dashboard_data(self, export_dir='output_data/dashboard_exports'):
        """
        Export ONLY JSON for Streamlit dashboard consumption. Ensures ISO dates.
        """
        try:
            # Ensure output directory exists
            os.makedirs(export_dir, exist_ok=True)
            export_path_json = os.path.join(export_dir, 'trading_data.json')

            # Normalize all dates → ISO string
            def iso_date(val):
                try:
                    ts = pd.to_datetime(val, errors='coerce')
                    return ts.strftime('%Y-%m-%d') if pd.notna(ts) else ''
                except Exception:
                    return str(val)

            export_data = {
                stock: [
                    {**row, 'Date': iso_date(row.get('Date', ''))}
                    for row in (data or []) if isinstance(row, dict)
                ]
                for stock, data in (self.stock_data or {}).items()
            }

            # Write JSON only
            with open(export_path_json, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"[OK] Dashboard data exported: {export_path_json}")
            return export_dir

        except Exception as e:
            print(f"❌ Error exporting dashboard data: {e}")
            return None


# Integration function to add to your main FilteringAndBacktesting class
def create_visualization_dashboard(backtested_scrips_df, backtested_transactions_df, launch=True):
    """
    Create and launch a visualization dashboard for the trading results.
    
    Args:
        backtested_scrips_df: DataFrame with OHLC data and signals
        backtested_transactions_df: DataFrame with transaction history  
        launch: Whether to automatically open the dashboard in browser
    
    Returns:
        TradingDashboard: Dashboard instance for further customization
    """
    dashboard = TradingDashboard(backtested_scrips_df, backtested_transactions_df)
    
    if launch:
        dashboard.launch_dashboard()
    
    return dashboard


# Add this method to your FilteringAndBacktesting class
def add_dashboard_method():
    """
    Method to add to your FilteringAndBacktesting class.
    """
    def create_dashboard(self, launch_browser=True):
        """
        Create and launch visualization dashboard for the trading results.
        
        Args:
            launch_browser: Whether to automatically open dashboard in browser
            
        Returns:
            TradingDashboard: Dashboard instance
        """
        if len(self.backtested_scrip_df_list) == 0:
            print("❌ No backtested data available. Run backtesting first.")
            return None
            
        # Combine all backtested data
        combined_scrips_df = pd.concat(self.backtested_scrip_df_list, ignore_index=True) if self.backtested_scrip_df_list else pd.DataFrame()
        combined_transactions_df = pd.concat(self.backtested_transactions_df_list, ignore_index=True) if self.backtested_transactions_df_list else pd.DataFrame()
        
        # Create dashboard
        dashboard = TradingDashboard(combined_scrips_df, combined_transactions_df)
        
        if launch_browser:
            dashboard.launch_dashboard()
        
        # Also export to dashboard_exports folder
        dashboard.export_dashboard_data()
        
        return dashboard
    
    return create_dashboard


# Modified run method for your FilteringAndBacktesting class
def enhanced_run_method():
    """
    Enhanced run method that includes dashboard generation.
    Add this to replace the existing run method in your FilteringAndBacktesting class.
    """
    def run(self, master_df, create_dashboard=True):
        """
        Complete portfolio management workflow with optional dashboard creation.
        
        Args:
            master_df: Master DataFrame with OHLC data
            create_dashboard: Whether to create visualization dashboard
            
        Returns:
            tuple: (backtested_scrips_df, backtested_transactions_df, dashboard)
        """
        print("🚀 STARTING PORTFOLIO MANAGEMENT SYSTEM")
        print(f"💰 Total Investment Capital: ₹{self.initial_cash:,.2f}")
        print(f"📊 Strategy: {config.ACTIVE_FILTER}")
        print(f"⏱️ Min Holding Period: {config.MIN_HOLDING_PERIOD} days")
        print(f"🎯 Min Profit Target: {config.MIN_PROFIT_PERCENTAGE}%")

        # Execute the complete workflow
        filtered_scrips_df = self.apply_filter(master_df)
        backtested_scrips_df, backtested_transactions_df = self.backtest_strategy(filtered_scrips_df)
        self.backtested_global_summary(backtested_scrips_df, backtested_transactions_df, master_df)

        dashboard = None
        if create_dashboard and not backtested_scrips_df.empty:
            print("\n" + "="*60)
            print("CREATING VISUALIZATION DASHBOARD".center(60))
            print("="*60)
            
            try:
                dashboard = TradingDashboard(backtested_scrips_df, backtested_transactions_df)
                dashboard_file = dashboard.launch_dashboard()
                
                if dashboard_file:
                    print(f"📊 Interactive Dashboard: {dashboard_file}")
                    print("🎯 Features Available:")
                    print("   • Interactive candlestick charts")
                    print("   • Buy/Sell signal markers with details")
                    print("   • Transaction history tables")
                    print("   • Portfolio performance statistics")
                    print("   • Zoom, pan, and annotation tools")
                
                # Export dashboard data
                dashboard.export_dashboard_data()
                
            except Exception as e:
                print(f"⚠️ Dashboard creation failed: {e}")
                print("📈 Trading analysis completed without visualization")

        return backtested_scrips_df, backtested_transactions_df, dashboard
    
    return run


# Example usage and integration instructions
if __name__ == "__main__":
    """
    Example of how to integrate the dashboard with your existing code.
    """
    
    # Example 1: Direct usage with your backtested data
    print("="*80)
    print("TRADING DASHBOARD INTEGRATION EXAMPLE".center(80))
    print("="*80)
    
    # Assuming you have your backtested data
    # backtested_scrips_df = pd.read_excel('backtested_scrips.xlsx')
    # backtested_transactions_df = pd.read_excel('backtested_transactions.xlsx')
    
    # Create and launch dashboard
    # dashboard = create_visualization_dashboard(
    #     backtested_scrips_df, 
    #     backtested_transactions_df, 
    #     launch=True
    # )
    
    print("""
    🎯 INTEGRATION INSTRUCTIONS:
    
    1. ADD TO YOUR FilteringAndBacktesting CLASS:
       
       from dashboard_integration import TradingDashboard
       
       def create_dashboard(self, launch_browser=True):
           '''Create visualization dashboard'''
           if len(self.backtested_scrip_df_list) == 0:
               print("No backtested data available")
               return None
               
           combined_scrips_df = pd.concat(self.backtested_scrip_df_list, ignore_index=True)
           combined_transactions_df = pd.concat(self.backtested_transactions_df_list, ignore_index=True)
           
           dashboard = TradingDashboard(combined_scrips_df, combined_transactions_df)
           if launch_browser:
               dashboard.launch_dashboard()
           return dashboard
    
    2. MODIFY YOUR RUN METHOD:
       
       def run(self, master_df, create_dashboard=True):
           # ... existing code ...
           backtested_scrips_df, backtested_transactions_df = self.backtest_strategy(filtered_scrips_df)
           self.backtested_global_summary(backtested_scrips_df, backtested_transactions_df, master_df)
           
           # Add dashboard creation
           dashboard = None
           if create_dashboard:
               dashboard = self.create_dashboard()
           
           return backtested_scrips_df, backtested_transactions_df, dashboard
    
    3. USAGE IN YOUR MAIN SCRIPT:
       
       portfolio_manager = FilteringAndBacktesting(initial_cash=100000.0)
       scrips_df, transactions_df, dashboard = portfolio_manager.run(master_df, create_dashboard=True)
    
    📊 DASHBOARD FEATURES:
    • Interactive candlestick charts with OHLC data
    • Buy/Sell signals marked as triangles on price timeline  
    • Hover tooltips showing trade details (quantity, price, P&L)
    • Portfolio statistics (win rate, total trades, avg holding period)
    • Transaction history table with filtering
    • Volume overlay option
    • Date range filtering
    • Zoom, pan, drawing tools
    • Export capabilities (PNG, HTML, JSON)
    
    💡 CUSTOMIZATION OPTIONS:
    • Add technical indicators overlay
    • Custom color schemes for different strategies
    • Multi-timeframe analysis
    • Portfolio comparison tools
    • Risk metrics visualization
    """)
    
    print("\n✅ Dashboard integration module ready!")
    print("📝 Save this as 'dashboard_integration.py' alongside your main script")
    print("🚀 Import and integrate with your FilteringAndBacktesting class")