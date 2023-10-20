# Domain shift
echo "PACS 0"
python main.py DomainBed/domainbed/data/ -d PACS           --task domain_shift --targets 0 --n-shot 4  -b 12 --lr 5e-6 --epochs 10 --lam 0.0 --beta 0.5 --log no_bma_lam0_pacs
echo "PACS 1"
python main.py DomainBed/domainbed/data/ -d PACS           --task domain_shift --targets 1 --n-shot 4  -b 12 --lr 5e-6 --epochs 10 --lam 0.0 --beta 0.5 --log no_bma_lam0_pacs
echo "PACS 2"
python main.py DomainBed/domainbed/data/ -d PACS           --task domain_shift --targets 2 --n-shot 4  -b 12 --lr 5e-6 --epochs 10 --lam 0.0 --beta 0.5 --log no_bma_lam0_pacs
echo "PACS 3"
python main.py DomainBed/domainbed/data/ -d PACS           --task domain_shift --targets 3 --n-shot 4  -b 12 --lr 5e-6 --epochs 10 --lam 0.0 --beta 0.5 --log no_bma_lam0_pacs
