# miRBase-tests

miRBase tests and utilities

## Notes

### BLAST

```
# BLAST for getting cblast file for MCL
sudo apt-get install ncbi-blast+-legacy
formatdb -i ../data/hairpin_db.fa -p F
blastall -i  ../data/hairpin_db.fa -d ../data/hairpin_db.fa -p blastn -m 8 > hairpin_m8.cblast
```

### MCL

```
sudo apt install mcl
# MCL "stream" execution, called within script each time
mcxdeblast --m9 --line-mode=abc --out=- ../data/hairpin_m8.cblast | mcl - --abc -o out.mcl
```