from data_processing import *

import yahoo_fin.stock_info as si
def get_eps(ticker):
    earn_hist = si.get_earnings_history(ticker)
    years_count = 20
    data_index = 4 + 4 * years_count

    eps = {}
    for d in earn_hist[4:data_index]:
        for k in ['startdatetime', 'epsactual', 'epsestimate', 'epssurprisepct']:
            print(k, d[k])
            if k == 'startdatetime':
                year = d[k][:4]
                #print(k, d[k][:10])
            elif k == 'epsactual':
                if year not in eps:
                    eps[year] = []
                eps[year].append(float(d[k]))
                if len(eps[year]) == 4:
                    year_eps = round(sum(eps[year]), 2)
                    print('\t', year, year_eps)
'''

#ge = si.get_earnings("aapl")
#pprint(ge)
#bs = si.get_balance_sheet(ticker)
#pprint(bs)


def get_avg_price(ticker, year):
    t1, t2 = '01/01/'+year, '12/21/'+year
    p_data = si.get_data(ticker, start_date = t1, end_date = t2)
    print(ticker, p_data['close'].mean())

def get_dividend(ticker):
    d = si.get_dividends(ticker , start_date = '01/01/2021' , end_date = '12/31/2021')
    print(ticker, d)

ticker = 'AAPL'

#get_eps(ticker)

get_avg_price(ticker, '2021')
get_dividend(ticker)

get_avg_price('GC=F', '2021')
get_avg_price('SI=F', '2021')

#income_statement = si.get_income_statement("amzn")
#balance_sheet = si.get_balance_sheet("amzn")
#cash_flow = si.get_cash_flow("amzn")

#print(income_statement)
#print('---')
#print(balance_sheet)
#print('---')
#print(cash_flow)

#quote = si.get_quote_table("aapl")
#print(quote)
'''

def main():
    print('------------------------------------------')
    symbol = 'AAPL'
    count = 0
    EPSs = get_EPS(symbol)
    APRs = get_APR(symbol)
    other_info = get_stock_indicators(symbol)
    ROEs = other_info['ROE']
    PEs = other_info['PE-Ratio']
    PBs = other_info['PB-Ratio']

    for i in range(5):
        #print(count, symbol, EPSs[i], APRs[i])
        print(count, symbol, EPSs[i], ROEs[i], PEs[i], PBs[i], APRs[i])
        count += 1


if __name__ == '__main__':
    main()
