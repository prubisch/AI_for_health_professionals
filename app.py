import seaborn as sns

# Import data from shared.py
# for modularisation we use the shiny core syntax
#from shared import df
from shiny import App, ui, render, reactive
from simple_AI_examples import simple_AI_page, server_AI_examples
from bias import bias_page, server_bias
from basics_AI import intro_page, intro_images
from bonus import bonus_page, bonus_images
from refs import refs_page


app_ui = ui.page_navbar(


    ui.nav_panel("Grundbausteine von KI", intro_page),
    ui.nav_panel("Einfach(st)e KI Beispiele", simple_AI_page),
    ui.nav_panel("Bias in den Trainingsdaten", bias_page), 
    ui.nav_panel("Bonus: Mathematische Grundlagen der Parameteroptimierung", bonus_page),
    ui.nav_panel("Referenzen", refs_page),
    title = "KI für Gesundheitsberufe", 
    position = 'static_top'
)


def server(input, output, session): 
    #all reactive and render functions go here
    intro_images(input)
    server_AI_examples(input)
    server_bias(input)
    bonus_images(input)
    


app = App(app_ui, server)