# ChinaAShareEquityCharacteristics
Equity return and characteristics of China A-Share market

- the codes are in `./code` folder, with full access
    - `./codemain.py` calculates stock characteristics
    - `./codemake_panel.ipynb` composes the panel data of stock returns and characteristics
    - `./code/sort_port/main_sort.py` makes sorted portfolios based on stock characteristics
    - `./code/sort_port/example.ipynb` output the returns and characteristics of the sorted portfolios
- all the data are in `./data` folder, not uploaded to the repository
    - `./chars/` stores the characteristics data, which are the output of `main.py`
    - `./factor/` stores the factors data, which are downloaded from other sources, for exmaple CH4 data from Robert Stambaugh's website
    - `./return` stores the monthly returns table from CSMAR
    - `./share` stores the output data of the repository, including the panel data of stock returns and characteristics, and sorted portfolio retunrs