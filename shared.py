#from pathlib import Path

#app_dir = Path(__file__).parent
#df = pd.read_csv(app_dir / "penguins.csv")


#if developing the style prefic needs to be changed, due to different local matplotlib version 
developing = False 
def seaborn_style(): 
    if developing: 
        return 'seaborn-'
    else: 
        return 'seaborn-v0_8-'
