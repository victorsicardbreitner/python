from operator import length_hint
from tkinter import *  
   
LaFenetre = Tk()  
LaFenetre.title("Manipuler Fichier")
LaFenetre.geometry("300x100")  

fichiersplit=""

def lecture() :
   global fichiersplit
   with open("C:/Users/Victor/Documents/Python/Manipuler Fichier/texte.txt", "r") as filin:
      fichier=filin.read()
   fichiersplit=fichier.split(".")

def ecriture() :
   with open("C:/Users/Victor/Documents/Python/Manipuler Fichier/texte2.txt", "w") as filout:
      for ligne in fichiersplit:
         texte=ligne+".\n"
         #texte=texte*12
         filout.write(texte)

boutonLecture = Button(LaFenetre, text ="Lecture", command = lecture)
boutonEcriture = Button(LaFenetre, text ="Ecriture", command = ecriture) 

label = Label(LaFenetre)  
#label.config(text = fichiersplit[0:5])
boutonLecture.pack() 
boutonEcriture.pack() 
label.pack()  
  
LaFenetre.mainloop()  