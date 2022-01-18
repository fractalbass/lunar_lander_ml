for f in *.csv; do
    mv -- "$f" "${f%.csv}_computer.csv"
done
