# Domain shift
echo "OfficeHome 0"
python main.py DomainBed/domainbed/data/ -d OfficeHome     --task domain_shift --targets 0 --n-shot 4  -b 12 --lr 1e-5 --epochs 10 --lam 0.0 --beta 0.5 --log no_bma_lam0_officehome
echo "OfficeHome 1"
python main.py DomainBed/domainbed/data/ -d OfficeHome     --task domain_shift --targets 1 --n-shot 4  -b 12 --lr 1e-5 --epochs 10 --lam 0.0 --beta 0.5 --log no_bma_lam0_officehome
echo "OfficeHome 2"
python main.py DomainBed/domainbed/data/ -d OfficeHome     --task domain_shift --targets 2 --n-shot 4  -b 12 --lr 1e-5 --epochs 10 --lam 0.0 --beta 0.5 --log no_bma_lam0_officehome
echo "OfficeHome 3"
python main.py DomainBed/domainbed/data/ -d OfficeHome     --task domain_shift --targets 3 --n-shot 4  -b 12 --lr 1e-5 --epochs 10 --lam 0.0 --beta 0.5 --log no_bma_lam0_officehome
