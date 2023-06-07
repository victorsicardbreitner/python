from operator import length_hint
from tkinter import *  
   
LaFenetre = Tk()  
LaFenetre.title("Manipuler Fichier")
LaFenetre.geometry("500x500")  

with open("C:/Users/Victor/Documents/Python/fichier.txt", "r") as filin:
   fichier=filin.read()
fichiersplit=fichier.split("\n")

with open("C:/Users/Victor/Documents/Python/fichier2.txt", "w") as filout:
   for ligne in fichiersplit:
      texte=ligne+"\n"
      texte=texte*12
      filout.write(texte)

label = Label(LaFenetre)  
label.config(text = fichiersplit[0:5])
label.pack()  
  
LaFenetre.mainloop()  