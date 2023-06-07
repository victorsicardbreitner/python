from tkinter import Toplevel, Button, Tk, Menu, Label  
  
LaFenetre = Tk()
LaFenetre.title("Les Menus")
LaFenetre.geometry("400x350")  

#afficher du texte dans la fenêtre
label = Label(LaFenetre)  
label.config(text = "Texte par défaut")
label.pack()

def fonctionNouveau():
    label.config(text = "Nouveau")

def fonctionOuvrir():
    label.config(text = "Ouvrir")

def fonctionGrandeFenetre():
    LaFenetre.geometry("500x400") 

def fonctionMoyenneFenetre():
    LaFenetre.geometry("400x350") 

def fonctionPetiteFenetre():
    LaFenetre.geometry("300x200") 

def fonctionAPropos():
    label.config(text = 
        "Les loutres (Lutrinae) sont une sous-famille de\n mammifères carnivores de la famille des mustélidés.\n Il existe plusieurs espèces de loutres, caractérisées\n par de courtes pattes, des doigts griffus et palmés\n (aux pattes avant et arrière) et une longue queue. ")

def fonctionQuiter():
    LaFenetre.quit()
    LaFenetre.destroy()

#définition d'une barre de menu
barreDeMenus = Menu(LaFenetre) 

#définition du menu file
fichier = Menu(barreDeMenus, tearoff=0)  
fichier.add_command(label="Nouveau", command=fonctionNouveau)  
fichier.add_command(label="Ouvrir", command=fonctionOuvrir)   
fichier.add_separator()  
fichier.add_command(label="Quiter", command=fonctionQuiter)  

#définition du menu format
format = Menu(barreDeMenus, tearoff=0)  
format.add_command(label="Grande Fenêtre", command=fonctionGrandeFenetre)
format.add_command(label="Moyenne Fenêtre", command=fonctionMoyenneFenetre) 
format.add_command(label="Petite Fenêtre", command=fonctionPetiteFenetre)   

#définition du menu help
aide = Menu(barreDeMenus, tearoff=0)  
aide.add_command(label="A propos", command=fonctionAPropos)  

 
#ajout des menus à la barre de menus
barreDeMenus.add_cascade(label="Fichier", menu=fichier) 
barreDeMenus.add_cascade(label="Format", menu=format)   
barreDeMenus.add_cascade(label="Aide", menu=aide)  




LaFenetre.config(menu=barreDeMenus)  
LaFenetre.mainloop()  




LaFenetre.mainloop()  



