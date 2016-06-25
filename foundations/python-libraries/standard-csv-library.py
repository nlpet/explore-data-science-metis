import csv

outfile = open('newfile.csv', 'w')
out = csv.writer(outfile, lineterminator='\n')
out.writerow(['header1', 'header2', 'header3', 'header4'])

for i in range(10):
    out.writerow([i, i+1, i+2, i+3])


outfile.close()
