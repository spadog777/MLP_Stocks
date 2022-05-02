from pprint import pprint
import os
import pandas as pd
import yahoo_fin.stock_info as si
import csv
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# companies: [symbol, company_name]
companies = [
    ['AAPL', 'apple'],
    ['GLPI', 'gaming-and-leisure-properties'],
    ['OKE', 'oneok'],
    ['UVV', 'universal'],
    ['PM', 'philip-morris'],
    ['VLO', 'valero-energy'],
    ['PSX', 'phillips-66'],
    ['EIX', 'edison'],
    ['NWE', 'northwestern'],
    ['ALE', 'allete'],
    ['SR', 'spire'],
    ['MMM', '3m'],
    ['LAMR', 'lamar-advertising'],
    ['WEC', 'wec-energy'],
    ['ES', 'eversource-energy'],
    ['CMI', 'cummins'],
    #['', ''],
    #['', ''],
]

# attribute: [attr, url_format)
attrs = [
    ['ROE', 'https://www.macrotrends.net/stocks/charts/{}/{}/roe'],
    ['PE-Ratio', 'https://www.macrotrends.net/stocks/charts/{}/{}/pe-ratio'],
    ['PB-Ratio', 'https://www.macrotrends.net/stocks/charts/{}/{}/price-book'],
    ['EPS', 'https://www.macrotrends.net/stocks/charts/{}/{}/eps-earnings-per-share-diluted'],
]

def get_APR(symbol):
    data = []
    years = ['2020', '2019', '2018', '2017', '2016']
    for year in years:
        t1, t2 = '01/01/'+year, '12/31/'+year
        p_data = si.get_data(symbol, start_date = t1, end_date = t2)
        avg_price = round(p_data['close'].mean(), 2)
        d = si.get_dividends(symbol, start_date = t1 , end_date = t2)
        apr = round(d['dividend'].sum() / avg_price * 100, 0)
        data.append(apr)

    return data

'''
def old_get_EPS(symbol):
    data = []
    earn_hist = si.get_earnings_history(symbol)
    years_count = 5
    data_index = 4 + 4 * years_count

    years = {'2020': [], '2019': [], '2018': [], '2017': [], '2016': []}
    for d in earn_hist[4:data_index]:
        for k in ['startdatetime', 'epsactual', 'epsestimate', 'epssurprisepct']:
            if k == 'startdatetime':
                year = d[k][:4]
            elif k == 'epsactual':
                years[year].append(float(d[k]))

    for year in years:
        total = round(sum(years[year]), 2)
        data.append(total)
    pprint(years)
    #pprint(data)

    return data
'''

def download_raw():
    for symbol, name in companies:
        print('downloading data of ', symbol)
        for attr, url_fmt in attrs:
            file_name = 'raw/{}_{}.csv'.format(symbol, attr)
            if os.path.isfile(file_name):
                continue
            url = url_fmt.format(symbol, name)
            tables = pd.read_html(url) # Returns list of all tables on page
            table = tables[0] # Select table of interest
            #print(table)
            table.to_csv(file_name)

def get_EPS(symbol):
    data = []
    attr, url_fmt = attrs[3]
    file_name = 'raw/{}_{}.csv'.format(symbol, attr)
    with open(file_name, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        # store past 5 five years of data
        years = {'2020': [], '2019': [], '2018': [], '2017': [], '2016': []}
        for row in reader:
            index, date, res = row
            year = date[:4]
            if year not in years:
                continue
            res = float(res[1:])
            data.append(res)
            #print(year, res)

    return data

def get_stock_indicators(symbol):
    dic = {
        'ROE': [],
        'PE-Ratio': [],
        'PB-Ratio': [],
    }

    for attr, url_fmt in attrs:
        # EPS file has different format, so I define another function get_EPS to parse data
        if attr == 'EPS':
            continue
        file_name = 'raw/{}_{}.csv'.format(symbol, attr)
        with open(file_name, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            # store past 5 five years of data
            years = {'2020': [], '2019': [], '2018': [], '2017': [], '2016': []}
            for row in reader:
                index, date, t1, t2, res = row
                year = date[:4]
                if year not in years:
                    continue
                res = float(res[:-1])
                years[year].append(res)

        #pprint(years)
        for year in years:
            avg_roe = round(sum(years[year]) / len(years[year]), 2)
            dic[attr].append(avg_roe)

    #pprint(dic)
    return dic

def generate_dataset():
    file_name = 'summary_summary.csv'
    with open(file_name, 'a+', newline='') as csvfile:
        # title
        row_list = ['', 'Symbol', 'EPS', 'ROE', 'PE', 'PB', 'APR']
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(row_list)
        count = 0
        for symbol, name in companies:
            EPSs = get_EPS(symbol)
            APRs = get_APR(symbol)
            other_info = get_stock_indicators(symbol)
            ROEs = other_info['ROE']
            PEs = other_info['PE-Ratio']
            PBs = other_info['PB-Ratio']

            # five records per company
            for i in range(5):
                try:
                    row_list = [count, symbol, EPSs[i], ROEs[i], PEs[i], PBs[i], APRs[i]]
                except:
                    print('exception i = ', i)
                    pprint(EPSs)
                    pprint(ROEs)
                    pprint(PEs)
                    pprint(PBs)
                    pprint(APRs)
                spamwriter.writerow(row_list)
                count += 1

def main():
    pass
    download_raw()
    generate_dataset()


if __name__ == '__main__':
    main()
