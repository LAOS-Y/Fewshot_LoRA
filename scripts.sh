# set -e

# Domain shift
echo "PACS 0"
python main.py DomainBed/domainbed/data/ -d PACS           --task domain_shift --targets 0 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
echo "PACS 1"
python main.py DomainBed/domainbed/data/ -d PACS           --task domain_shift --targets 1 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
echo "PACS 2"
python main.py DomainBed/domainbed/data/ -d PACS           --task domain_shift --targets 2 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
echo "PACS 3"
python main.py DomainBed/domainbed/data/ -d PACS           --task domain_shift --targets 3 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5

echo "VLCS 0"
python main.py DomainBed/domainbed/data/ -d VLCS           --task domain_shift --targets 0 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
echo "VLCS 1"
python main.py DomainBed/domainbed/data/ -d VLCS           --task domain_shift --targets 1 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
echo "VLCS 2"
python main.py DomainBed/domainbed/data/ -d VLCS           --task domain_shift --targets 2 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
echo "VLCS 3"
python main.py DomainBed/domainbed/data/ -d VLCS           --task domain_shift --targets 3 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5

echo "OfficeHome 0"
python main.py DomainBed/domainbed/data/ -d OfficeHome     --task domain_shift --targets 0 -b 12 --lr 1e-5 --epochs 10 --lam 0.3 --beta 0.5
echo "OfficeHome 1"
python main.py DomainBed/domainbed/data/ -d OfficeHome     --task domain_shift --targets 1 -b 12 --lr 1e-5 --epochs 10 --lam 0.3 --beta 0.5
echo "OfficeHome 2"
python main.py DomainBed/domainbed/data/ -d OfficeHome     --task domain_shift --targets 2 -b 12 --lr 1e-5 --epochs 10 --lam 0.3 --beta 0.5
echo "OfficeHome 3"
python main.py DomainBed/domainbed/data/ -d OfficeHome     --task domain_shift --targets 3 -b 12 --lr 1e-5 --epochs 10 --lam 0.3 --beta 0.5

echo "TerraIncognita 0"
python main.py DomainBed/domainbed/data/ -d TerraIncognita --task domain_shift --targets 0 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
echo "TerraIncognita 1"
python main.py DomainBed/domainbed/data/ -d TerraIncognita --task domain_shift --targets 1 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
echo "TerraIncognita 2"
python main.py DomainBed/domainbed/data/ -d TerraIncognita --task domain_shift --targets 2 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
echo "TerraIncognita 3"
python main.py DomainBed/domainbed/data/ -d TerraIncognita --task domain_shift --targets 3 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5

echo "DomainNet 0"
python main.py DomainBed/domainbed/data/ -d DomainNet      --task domain_shift --targets 0 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5
echo "DomainNet 1"
python main.py DomainBed/domainbed/data/ -d DomainNet      --task domain_shift --targets 1 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5
echo "DomainNet 2"
python main.py DomainBed/domainbed/data/ -d DomainNet      --task domain_shift --targets 2 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5
echo "DomainNet 3"
python main.py DomainBed/domainbed/data/ -d DomainNet      --task domain_shift --targets 3 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5
echo "DomainNet 4"
python main.py DomainBed/domainbed/data/ -d DomainNet      --task domain_shift --targets 4 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5
echo "DomainNet 5"
python main.py DomainBed/domainbed/data/ -d DomainNet      --task domain_shift --targets 5 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5

# # Open class
# python main.py CoOp/data/ -d ImageNet            --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.1
# python main.py CoOp/data/ -d Caltech101          --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# python main.py CoOp/data/ -d OxfordPets          --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# python main.py CoOp/data/ -d StanfordCars        --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 5  --lam 0.3 --beta 0.1
# python main.py CoOp/data/ -d OxfordFlowers       --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# python main.py CoOp/data/ -d Food101             --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# python main.py CoOp/data/ -d FGVCAircraft        --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.1
# python main.py CoOp/data/ -d SUN397              --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 5  --lam 0.3 --beta 0.1
# python main.py CoOp/data/ -d DescribableTextures --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# python main.py CoOp/data/ -d EuroSAT             --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# python main.py CoOp/data/ -d UCF101              --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 2  --lam 0.3 --beta 0.1

# # In-the-wild
# python main.py DomainBed/domainbed/data/ -d OfficeHome --task in_the_wild --targets 0 -b 12 --lr 3e-6 --epochs 10 --lam 0.1 --beta 0.1
# python main.py DomainBed/domainbed/data/ -d OfficeHome --task in_the_wild --targets 1 -b 12 --lr 3e-6 --epochs 10 --lam 0.1 --beta 0.1
# python main.py DomainBed/domainbed/data/ -d OfficeHome --task in_the_wild --targets 2 -b 12 --lr 3e-6 --epochs 10 --lam 0.1 --beta 0.1
# python main.py DomainBed/domainbed/data/ -d OfficeHome --task in_the_wild --targets 3 -b 12 --lr 3e-6 --epochs 10 --lam 0.1 --beta 0.1

# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 0 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5
# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 1 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5
# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 2 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5
# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 3 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5
# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 4 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5
# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 5 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5