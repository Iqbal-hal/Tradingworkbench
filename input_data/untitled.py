
import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg
from matplotlib.figure import Figure
import talib

# ======================
# Global toggles
# ======================
main_plot_enabled = True     # Main chart (close price, EMA, Bollinger, etc.)
bollinger_enabled = True
ema_9_enabled = True
ema_20_enabled = True
pe_enabled = True
annotation_text_enabled = True
close_price_enabled = True
axv_line_enabled = True

# New globals for dynamic font sizes:
annotation_fontsize = 10     # Default annotation text font size
annotation_orient = 7
limt_multiplier = 1.1
xaxis_label_fontsize = 8     # Default x-axis tick label font size

# New toggles for the extra indicator plots:
macd_enabled = False         # Toggle for MACD (MACD, Signal, Histogram)
rsi_enabled = False          # Toggle for RSI
atr_enabled = False          # Toggle for ATR

def annotation(df, df_scrip_gain): 
    # --- Annotations (buy, sell, etc.) ---
    buy_date   = df_scrip_gain['Buy Date']
    buy_price  = df_scrip_gain['Buy Price']
    buy_rsi    = df.loc[buy_date, 'RSI'] if buy_date in df.index else None
    buy_pe     = df.loc[buy_date, 'P/E'] if (buy_date in df.index and 'P/E' in df.columns) else None

    sell_date  = df_scrip_gain['Sell Date']
    sell_price = df_scrip_gain['Sell Price']
    sell_rsi   = df.loc[sell_date, 'RSI'] if sell_date in df.index else None
    sell_pe    = df.loc[sell_date, 'P/E'] if (sell_date in df.index and 'P/E' in df.columns) else None

    min_date   = df_scrip_gain['Min Date']
    min_price  = df_scrip_gain['Min Price']
    min_rsi    = df.loc[min_date, 'RSI'] if min_date in df.index else None
    min_pe     = df.loc[min_date, 'P/E'] if (min_date in df.index and 'P/E' in df.columns) else None

    max_date   = df_scrip_gain['Max Date']
    max_price  = df_scrip_gain['Max Price']
    max_rsi    = df.loc[max_date, 'RSI'] if max_date in df.index else None
    max_pe     = df.loc[max_date, 'P/E'] if (max_date in df.index and 'P/E' in df.columns) else None

    trade_gain     = df_scrip_gain['Trade Gain']
    max_gain       = df_scrip_gain['Max Gain']
    max_gain_signal= df_scrip_gain['Signal Max Gain']

    min_date_signal  = df_scrip_gain['Signal Min Date']
    min_price_signal = df_scrip_gain['Signal Min Price']
    min_rsi_signal   = df.loc[min_date_signal, 'RSI'] if min_date_signal in df.index else None
    min_pe_signal    = df.loc[min_date_signal, 'P/E'] if (min_date_signal in df.index and 'P/E' in df.columns) else None

    max_date_signal  = df_scrip_gain['Signal Max Date']
    max_price_signal = df_scrip_gain['Signal Max Price']
    max_rsi_signal   = df.loc[max_date_signal, 'RSI'] if max_date_signal in df.index else None
    max_pe_signal    = df.loc[max_date_signal, 'P/E'] if (max_date_signal in df.index and 'P/E' in df.columns) else None

    dates = [buy_date, sell_date, min_date, max_date, min_date_signal, max_date_signal]
    prices = [buy_price, sell_price, min_price, max_price, min_price_signal, max_price_signal]
    rsi_values = [buy_rsi, sell_rsi, min_rsi, max_rsi, min_rsi_signal, max_rsi_signal]
    pe_values  = [buy_pe, sell_pe, min_pe, max_pe, min_pe_signal, max_pe_signal]
    gains = [trade_gain, trade_gain, max_gain, max_gain, max_gain_signal, max_gain_signal]
    notes = ['Buy', 'Sell', 'Min', 'Max', 'MinSig', 'MaxSig']
    styles = ['--', '--', '-.', '-.', ':', ':']
    textcolors = ['', '', '', '', '', '']

    if buy_date < sell_date:
        b = 'green'
        s = 'red'
        notes[0] = ' (1,Sig_ind,B,V)'
        notes[1] = ' (1,Sig_ind,S,V)'
        textcolors[0] = 'green'
        textcolors[1] = 'red'
    else:
        b = 'green'
        s = 'red'
        notes[0] = ' (1,Sig_ind,B,InV)'
        notes[1] = ' (1,Sig_ind,S,InV)'
        textcolors[0] = 'magenta'
        textcolors[1] = 'magenta'

    if min_date < max_date:
        min_ = 'green'
        max_ = 'red'
        notes[2] = ' (2,Pd_min,B,V)'
        notes[3] = ' (2,Pd_max,S,V)'
        textcolors[2] = 'green'
        textcolors[3] = 'red'
    else:
        min_ = 'green'
        max_ = 'red'
        notes[2] = ' (2,Pd_min,B,InV)'
        notes[3] = ' (2,Pd_max,S,InV)'
        textcolors[2] = 'magenta'
        textcolors[3] = 'magenta'

    if min_date_signal < max_date_signal:
        min_sig = 'green'
        max_sig = 'red'
        notes[4] = ' (3,Sig_Min,B,V)'
        notes[5] = ' (3,Sig_Max,S,V)'
        textcolors[4] = 'green'
        textcolors[5] = 'red'
    else:
        min_sig = 'green'
        max_sig = 'red'
        notes[4] = ' (3,Sig_Min,B,InV)'
        notes[5] = ' (3,Sig_Max,S,InV)'
        textcolors[4] = 'magenta'
        textcolors[5] = 'magenta'

    linecolors = [b, s, min_, max_, min_sig, max_sig]
    lim_max = df['Close'].max() * limt_multiplier
    delta = df['Close'].max() / annotation_orient
    positions = [delta, 2.5 * delta, 3.5 * delta, 4.5 * delta, 5.5 * delta, 6.5 * delta]

    return notes, dates, prices, rsi_values, pe_values, gains, linecolors, textcolors, styles, positions, lim_max

    
# ======================
# Plotting functions
# ======================

def plot_main(ax, volume_ax, scrip, df, df_scrip_gain):
    """
    Plot the main chart with close price, Bollinger bands, EMA lines, PE ratio and volume.
    (The indicator calculations are assumed to have been computed already in df.)
    """
    ax.clear()
    volume_ax.clear()

    if close_price_enabled:
        ax.plot(df.index, df['Close'], label='Close Price', color='navy', alpha=0.8, linestyle='-')

    if bollinger_enabled:
        ax.plot(df.index, df['Bollinger_Upper'], label='Bollinger Upper', color='orange', alpha=0.6, linestyle='--')
        ax.plot(df.index, df['Bollinger_Middle'], label='Bollinger Middle', color='gray', alpha=0.5, linestyle='-.')
        ax.plot(df.index, df['Bollinger_Lower'], label='Bollinger Lower', color='purple', alpha=0.6, linestyle=':')

    if ema_9_enabled:
        ax.plot(df.index, df['ema_9'], label='EMA 9', color='lime', alpha=0.7, linestyle='-')

    if ema_20_enabled:
        ax.plot(df.index, df['ema_20'], label='EMA 20', color='cyan', alpha=0.7, linestyle='--')

    if pe_enabled and 'P/E' in df.columns:
        ax.plot(df.index, df['P/E'], label='P/E', color='magenta', alpha=0.6, linestyle='-.')

    notes, dates, prices, rsi_values, pe_values, gains, linecolors, textcolors, styles, positions, lim_max = annotation(df, df_scrip_gain) 

    for note, date, price, rsi_val, pe_val, gain, lcolor, tcolor, style, pos in zip(
            notes, dates, prices, rsi_values, pe_values, gains, linecolors, textcolors, styles, positions):
        annotation_text = f"  {note}\n  {date}\n   Price:{price}\n   RSI:{rsi_val}\n   PE:{pe_val}\n   Gain:{gain}"
        if axv_line_enabled:
            ax.axvline(x=date, color=lcolor, linestyle=style, linewidth=2)
        if annotation_text_enabled:
            ax.text(date, pos, annotation_text, fontsize=annotation_fontsize, color=tcolor)

    ax.set_title(scrip)
    ax.set_ylabel('Price', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=xaxis_label_fontsize)
    ax.set_ylim(0, lim_max)

    volume = df['Volume']
    close_prices = df['Close']
    colors = ['g' if close_prices.iloc[i] > close_prices.iloc[i - 1] else 'r'
              for i in range(1, len(close_prices))]
    colors.insert(0, 'g')
    volume_ax.bar(df.index, volume, color=colors, alpha=0.4)
    volume_ax.plot(df.index, df['volume_ema_20'], label='Volume EMA 20', color='blue', alpha=0.5)
    volume_ax.set_ylabel('Volume', color='black')
    volume_ax.yaxis.tick_right()
    volume_ax.tick_params(axis='y', labelcolor='black')
    volume_max = volume.max() * 2
    volume_ax.set_ylim(0, volume_max)

def plot_macd(ax, scrip, df, df_scrip_gain):
    notes, dates, prices, rsi_values, pe_values, gains, linecolors, textcolors, styles, positions, lim_max = annotation(df, df_scrip_gain)
    hist_diff = df['Hist'].diff()

    def set_hist_dif_alpha(row, diff):
        if row.Hist > 0:
            return 0.9 if diff > 0 else 0.5
        else:
            return 0.4 if diff > 0 else 0.8

    df['Hist_dif_alpha'] = [set_hist_dif_alpha(row, diff) for row, diff in zip(df.itertuples(), hist_diff)]
    colors = ['g' if val >= 0 else 'r' for val in df['Hist']]
    alphas = df['Hist_dif_alpha']    
    
    ax.clear()
    ax.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax.plot(df.index, df['Signal'], label='Signal', color='orange')
    for date, lcolor, style in zip(dates, linecolors, styles):        
        ax.axvline(x=date, color=lcolor, linestyle=style, linewidth=1)
    bars = ax.bar(df.index, df['Hist'], color=colors)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    
    ax.set_title(f'MACD for {scrip}')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=xaxis_label_fontsize)

def plot_rsi(ax, scrip, df):
    ax.clear()
    ax.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax.axhline(30, color='green', linestyle='--', linewidth=1)
    ax.axhline(70, color='red', linestyle='--', linewidth=1)
    ax.set_title(f'RSI for {scrip}')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=xaxis_label_fontsize)

def plot_atr(ax, scrip, df):
    ax.clear()
    ax.plot(df.index, df['ATR'], label='ATR', color='brown')
    ax.set_title(f'ATR for {scrip}')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=xaxis_label_fontsize)


# ======================
# iterate_list function with dynamic layout and horizontal scroll slider
# ======================

def iterate_list(filtered_ohlc_gain_df):
    """
    This function iterates through the stocks and updates the plot.
    It builds a dynamic layout using GridSpec for the enabled plots.
    Now it also adds a horizontal slider that lets you â€œscrollâ€ through the data.
    """
    global main_plot_enabled, bollinger_enabled, ema_9_enabled, ema_20_enabled, pe_enabled, close_price_enabled, axv_line_enabled
    global macd_enabled, rsi_enabled, atr_enabled, annotation_text_enabled

    filtered_ohlc_df = filtered_ohlc_gain_df[0]
    filtered_gain_df = filtered_ohlc_gain_df[1]
    stock_list = filtered_ohlc_df['Stock'].unique().tolist()
    current_stock_index = 0

    root = tk.Tk()
    root.wm_title("Stock Plot")

    container = tk.Frame(root)
    container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    fig = Figure(figsize=(16, 12), dpi=120, constrained_layout=True)
    canvas = FigureCanvasTkAgg(fig, master=container)

    # Variable to hold the currently plotted dataframe (used for scrolling)
    current_df = None

    # Define a function that updates the x-axis limits based on the scroll slider.
    def update_scroll(val=None):
        nonlocal current_df
        if current_df is None:
            return
        try:
            offset = float(scroll_slider.get()) if val is None else float(val)
        except Exception:
            offset = 0
        start_date = current_df.index[0]
        end_date = current_df.index[-1]
        total_days = (end_date - start_date).days
        window_days = 365  # adjust this to set the visible time window (e.g. 1 year)
        if total_days > window_days:
            delta_days = total_days - window_days
            offset_days = int(offset * delta_days)
            new_start = start_date + pd.Timedelta(days=offset_days)
            new_end = new_start + pd.Timedelta(days=window_days)
        else:
            new_start = start_date
            new_end = end_date
        for ax in fig.get_axes():
            ax.set_xlim(new_start, new_end)
        canvas.draw_idle()

    # Custom Toolbar with your buttons and sliders
    class CustomToolbar(NavigationToolbar2Tk):
        def __init__(self, canvas, window):
            super().__init__(canvas, window)
            separator = ttk.Separator(self, orient='vertical')
            separator.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
            
            self.toggle_plot_button = tk.Button(master=self, text="Main Plot", command=self.toggle_main_plot)
            self.toggle_plot_button.pack(side=tk.LEFT)
            
            self.toggle_macd_button = tk.Button(master=self, text="MACD", command=self.toggle_macd)
            self.toggle_macd_button.pack(side=tk.LEFT)
            
            self.toggle_rsi_button = tk.Button(master=self, text="RSI", command=self.toggle_rsi)
            self.toggle_rsi_button.pack(side=tk.LEFT)
    
            self.toggle_atr_button = tk.Button(master=self, text="ATR", command=self.toggle_atr)
            self.toggle_atr_button.pack(side=tk.LEFT)
    
            self.toggle_close_price_button = tk.Button(master=self, text="Close Price", command=self.toggle_close_price)
            self.toggle_close_price_button.pack(side=tk.LEFT)
            
            self.toggle_bollinger_button = tk.Button(master=self, text="Bollinger Bands", command=self.toggle_bollinger)
            self.toggle_bollinger_button.pack(side=tk.LEFT)
            
            self.toggle_ema_9_button = tk.Button(master=self, text="EMA 9", command=self.toggle_ema_9)
            self.toggle_ema_9_button.pack(side=tk.LEFT)
            
            self.toggle_ema_20_button = tk.Button(master=self, text="EMA 20", command=self.toggle_ema_20)
            self.toggle_ema_20_button.pack(side=tk.LEFT)
            
            self.toggle_pe_button = tk.Button(master=self, text="PE Ratio", command=self.toggle_pe)
            self.toggle_pe_button.pack(side=tk.LEFT)
            
            self.toggle_annotation_button = tk.Button(master=self, text="Text", command=self.toggle_annotation)
            self.toggle_annotation_button.pack(side=tk.LEFT)
    
            self.toggle_axvline_button = tk.Button(master=self, text="X-Line", command=self.toggle_axv_line)
            self.toggle_axvline_button.pack(side=tk.LEFT)
    
            separator2 = ttk.Separator(self, orient='vertical')
            separator2.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
    
            self.annotation_slider = tk.Scale(self, from_=5, to=30, orient=tk.HORIZONTAL, label="", command=self.update_annotation_fontsize)
            self.annotation_slider.set(annotation_fontsize)
            self.annotation_slider.pack(side=tk.LEFT, padx=5)
    
            self.xaxis_slider = tk.Scale(self, from_=5, to=30, orient=tk.HORIZONTAL, label="", command=self.update_xaxis_fontsize)
            self.xaxis_slider.set(xaxis_label_fontsize)
            self.xaxis_slider.pack(side=tk.LEFT, padx=5)
    
            separator3 = ttk.Separator(self, orient='vertical')
            separator3.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
    
            self.annotation_orientation_slider = tk.Scale(self, from_=4, to=9, orient=tk.HORIZONTAL, label="", command=self.update_annotaion_orientation)
            self.annotation_orientation_slider.set(annotation_orient)
            self.annotation_orientation_slider.pack(side=tk.LEFT, padx=5)
    
            self.limit_multiplier_slider = tk.Scale(self, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="", command=self.update_limit_multiplier)
            self.limit_multiplier_slider.set(limt_multiplier)
            self.limit_multiplier_slider.pack(side=tk.LEFT)
    
        def toggle_close_price(self):
            global close_price_enabled
            close_price_enabled = not close_price_enabled
            update_plot()
    
        def toggle_bollinger(self):
            global bollinger_enabled
            bollinger_enabled = not bollinger_enabled
            update_plot()
    
        def toggle_ema_9(self):
            global ema_9_enabled
            ema_9_enabled = not ema_9_enabled
            update_plot()
    
        def toggle_ema_20(self):
            global ema_20_enabled
            ema_20_enabled = not ema_20_enabled
            update_plot()
    
        def toggle_pe(self):
            global pe_enabled
            pe_enabled = not pe_enabled
            update_plot()
    
        def toggle_annotation(self):
            global annotation_text_enabled
            annotation_text_enabled = not annotation_text_enabled
            update_plot()
    
        def toggle_axv_line(self):
            global axv_line_enabled
            axv_line_enabled = not axv_line_enabled
            update_plot()
            
        def toggle_main_plot(self):
            global main_plot_enabled
            main_plot_enabled = not main_plot_enabled
            update_plot()
    
        def toggle_macd(self):
            global macd_enabled
            macd_enabled = not macd_enabled
            update_plot()
    
        def toggle_rsi(self):
            global rsi_enabled
            rsi_enabled = not rsi_enabled
            update_plot()
    
        def toggle_atr(self):
            global atr_enabled
            atr_enabled = not atr_enabled
            update_plot()
    
        def update_annotation_fontsize(self, val):
            global annotation_fontsize
            annotation_fontsize = int(val)
            update_plot()
    
        def update_xaxis_fontsize(self, val):
            global xaxis_label_fontsize
            xaxis_label_fontsize = int(val)
            update_plot()
    
        def update_annotaion_orientation(self, val):
            global annotation_orient
            annotation_orient = float(val)
            update_plot()
    
        def update_limit_multiplier(self, val):
            global limt_multiplier
            limt_multiplier = float(val)
            update_plot()

    toolbar = CustomToolbar(canvas, container)
    toolbar.pack(side=tk.TOP, fill=tk.X)
    toolbar.update()    
    
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)   
    
    # Create a frame for the horizontal scroll slider at the bottom.
    scroll_frame = tk.Frame(root)
    scroll_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    scroll_slider = tk.Scale(scroll_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,
                             label="Scroll Years", command=update_scroll)
    scroll_slider.set(0)
    scroll_slider.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    
    def update_plot():
        nonlocal current_stock_index, current_df
        current_stock = stock_list[current_stock_index]
        df_scrip_ohlc = filtered_ohlc_df[filtered_ohlc_df['Stock'] == current_stock].copy()
        df_scrip_gain = filtered_gain_df.loc[current_stock, :]
        df = df_scrip_ohlc.copy()
        df['ema_200'] = talib.EMA(df['Close'], timeperiod=200)
        df['ema_100'] = talib.EMA(df['Close'], timeperiod=100)
        df['ema_50']  = talib.EMA(df['Close'], timeperiod=50)
        df['ema_30']  = talib.EMA(df['Close'], timeperiod=30)
        df['ema_20']  = talib.EMA(df['Close'], timeperiod=20)
        df['ema_9']   = talib.EMA(df['Close'], timeperiod=9)
        df['volume_ema_20'] = talib.EMA(df['Volume'], timeperiod=20)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14).round(2)
        df['MACD'], df['Signal'], df['Hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        n = 20
        std_dev = 2
        df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(
            df['Close'], timeperiod=n, nbdevup=std_dev, nbdevdn=std_dev, matype=0
        )
    
        plots = []
        if main_plot_enabled:
            plots.append('main')
        if macd_enabled:
            plots.append('macd')
        if rsi_enabled:
            plots.append('rsi')
        if atr_enabled:
            plots.append('atr')
            
        n_plots = len(plots)
        if n_plots == 0:
            return
            
        fig.clear()
        if main_plot_enabled:
            ratios = [3] + [1]*(n_plots - 1)
        else:
            ratios = [1] * n_plots
        gs = fig.add_gridspec(n_plots, 1, height_ratios=ratios, hspace=0.05)
        for i, plot_key in enumerate(plots):
            ax = fig.add_subplot(gs[i, 0])
            if plot_key == 'main':
                volume_ax = ax.twinx()
                plot_main(ax, volume_ax, current_stock, df, df_scrip_gain)
            elif plot_key == 'macd':
                plot_macd(ax, current_stock, df, df_scrip_gain)
            elif plot_key == 'rsi':
                plot_rsi(ax, current_stock, df)
            elif plot_key == 'atr':
                plot_atr(ax, current_stock, df)
        current_df = df  # store the current dataframe for scrolling purposes
        canvas.draw_idle()
        update_scroll()  # update the x-axis range based on the current slider value
    
    def on_key(event):
        nonlocal current_stock_index
        if event.keysym == 'Right':
            current_stock_index = (current_stock_index + 1) % len(stock_list)
            update_plot()
        elif event.keysym == 'Left':
            current_stock_index = (current_stock_index - 1) % len(stock_list)
            update_plot()
    
    root.bind('<Key>', on_key)
    update_plot()  # Initial plot
    tk.mainloop()

