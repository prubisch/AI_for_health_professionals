from shiny import ui,render
from shiny.types import ImgData


intro_page = ui.page_fluid(
    ui.HTML("<p>Systeme in der Künstlichen Intelligenz oder auch Machine Learning Algorithmen können auf 4 grundlegende Bausteine reduziert werden:</p>"),
    ui.HTML("<ul><li>Objective</li><li>Trainings- und Testdaten</li><li>Architektur</li><li>Trainingsalgorithmus</></ul>"),
    ui.HTML("<p>Vereinfacht kann man ein Intelligentes System für das Beispiel der Klassifizierung so darstellen:</p>"),
    ui.output_image("overview", fill = True),
    ui.HTML("<p style='margin-top: -200px;'>Wir fokussieren uns auf den nächsten Seiten auf die Objektive Funktion und den Einfluss der Trainingsdaten. Die Funktionsweise wird anhand von Beispielen und Visualisierungen erklärt. Falls Sie sich für die mathematischen Details der Optimierung und Definition des Objectives interessieren, können Sie sich hierzu weitere Details auf der <i>Bonusseite</i> finden.</p>"),
    
)

def intro_images(input): 

    @render.image
    def overview():
        from pathlib import Path
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": dir / "overview.png"}
        return img