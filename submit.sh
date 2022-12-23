echo 'calculate stock-level characteristics \n'
python ./code/main.py > ./code/main.log.txt 2>&1

echo 'compose the panel of stock returns and characteristics \n'
python ./code/make_panel.py > ./code/make_panel.log.txt 2>&1

echo 'portfolios sorting \n'
python ./code/sort_port/main_sort.py > ./code/sort_port/main_sort.log.txt 2>&1
python ./code/sort_port/main_sort_example.py > ./code/sort_port/main_sort_example.log.txt 2>&1

echo 'finished \n'