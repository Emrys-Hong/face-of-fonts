import tkinter as tk
from PIL import ImageTk,Image
import CNN_test_accuracy
from font_recommender import Vector_composition

def recommendations_plus_matching():
    global canvas1
    global window
    mission = v.get()
    ideal_font = v2.get()
    dictionary = CNN_test_accuracy.CNN.score(mission)
    #1. find the input
    helvetica_neue = dictionary['Helvetica Nueue']
    helvetica = dictionary['Helvetica']
    arial = dictionary['Arial']
    myriad = dictionary['Myriad']
    futura = dictionary['Futura']
    gotham = dictionary['Gotham']
    frutiger = dictionary['Frutiger']
    gill_sans = dictionary['Gill Sans']
    garamond = dictionary['Garamond']
    #2. do the vector composition, and find the largest proportion
    font_vector = Vector_composition(helvetica_neue, helvetica, arial, myriad, futura, gotham, frutiger, gill_sans,
                                     garamond)
    font_vector.vector_composition()
    font = font_vector.found_closest_font_cosine()
    font_recommender = tk.Label(window,text=font[1][0], font=('Futura', 30)).place(x=950,y=120)

    #compare the similarity:
    tk.Label(window,text='cosine similarity:').place(x=900,y=190)
    if ideal_font == 'Helvetica Neue':
        value = font_vector.find_cosine_similarity("Roboto regular")
        text = tk.Label(window,text=value,font=('Futura',13)).place(x=1000,y=170)
    elif ideal_font == 'Helvetica':
        value = font_vector.find_cosine_similarity("Doppio One regular")
        text = tk.Label(window, text=value, font=('Futura', 13)).place(x=1000, y=170)
    elif ideal_font == 'Arial':
        value = font_vector.find_cosine_similarity("Shanti regular")
        text = tk.Label(window, text=value, font=('Futura', 13)).place(x=1000, y=170)
    elif ideal_font == 'Myriad':
        value = font_vector.find_cosine_similarity("PT Sans regular")
        text = tk.Label(window, text=value, font=('Futura', 13)).place(x=1000, y=170)
    elif ideal_font == 'Futura':
        value = font_vector.find_cosine_similarity("Nunito regular")
        text = tk.Label(window, text=value, font=('Futura', 13)).place(x=1000, y=170)
    elif ideal_font == 'Gotham':
        value = font_vector.find_cosine_similarity("Finger Paint regular")
        text = tk.Label(window, text=value, font=('Futura', 13)).place(x=1000, y=170)
    elif ideal_font == 'Frutiger':
        value = font_vector.find_cosine_similarity("Istok Web regular")
        text = tk.Label(window, text=value, font=('Futura', 13)).place(x=1000, y=170)
    elif ideal_font == 'Gill Sans':
        value = font_vector.find_cosine_similarity("Oxygen regular")
        text = tk.Label(window, text=value, font=('Futura', 13)).place(x=1000, y=170)
    elif ideal_font == 'Garamond':
        value = font_vector.find_cosine_similarity("EB Garamond regular")
        text = tk.Label(window, text=value, font=('Futura', 13)).place(x=1000, y=170)

    #analysing result
    tk.Label(window,text='Helvetica Nueue',font=('Futura',13)).place(x=820,y=300)
    tk.Label(window,text=dictionary['Helvetica Nueue'],font=('Futura',13)).place(x=940,y=300)
    tk.Label(window,text='Helvetica',font=('Futura',13)).place(x=1010,y=300)
    tk.Label(window,text=dictionary['Helvetica'],font=('Futura',13)).place(x=1120,y=300)

    tk.Label(window,text='Arial', font=('Futura', 13)).place(x=820, y=320)
    tk.Label(window,text=dictionary['Arial'], font=('Futura', 13)).place(x=930, y=320)
    tk.Label(window,text='Myriad', font=('Futura', 13)).place(x=1000, y=320)
    tk.Label(window,text=dictionary['Myriad'], font=('Futura', 13)).place(x=1110, y=320)

    tk.Label(window,text='Futura', font=('Futura', 13)).place(x=820, y=340)
    tk.Label(window,text=dictionary['Futura'], font=('Futura', 13)).place(x=930, y=340)
    tk.Label(window,text='Gotham', font=('Futura', 13)).place(x=1000, y=340)
    tk.Label(window,text=dictionary['Gotham'], font=('Futura', 13)).place(x=1110, y=340)

    tk.Label(window,text='Frutiger', font=('Futura', 13)).place(x=820, y=360)
    tk.Label(window,text=dictionary['Frutiger'], font=('Futura', 13)).place(x=930, y=360)
    tk.Label(window,text='Gill Sans', font=('Futura', 13)).place(x=1000, y=360)
    tk.Label(window,text=dictionary['Gill Sans'], font=('Futura', 13)).place(x=1110, y=360)

    tk.Label(window,text='Garamond', font=('Futura', 13)).place(x=820, y=380)
    tk.Label(window,text=dictionary['Garamond'], font=('Futura', 13)).place(x=930, y=380)

    #main font, find the font
    largest_proportion = max(dictionary, key=dictionary.get)
    if largest_proportion == 'Helvetica Nueue':
        tk.Label(window, text='Helvetica Nueue', font=('Helvetica Nueue', 30)).place(x=950, y=500)
    elif largest_proportion == 'Helvetica':
        tk.Label(window, text='Helvetica', font=('Helvetica', 30)).place(x=950, y=500)
    elif largest_proportion == 'Arial':
        tk.Label(window, text='Arial', font=('Arial', 30)).place(x=950, y=500)
    elif largest_proportion == 'Myriad':
        tk.Label(window, text='Myriad', font=('Myriad', 30)).place(x=950, y=500)
    elif largest_proportion == 'Futura':
        tk.Label(window, text='Futura', font=('Futura', 30)).place(x=950, y=500)
    elif largest_proportion == 'Gotham':
        tk.Label(window, text='Gotham', font=('Gotham', 30)).place(x=950, y=500)
    elif largest_proportion == 'Frutiger':
        tk.Label(window, text='Frutiger', font=('Frutiger', 30)).place(x=950, y=500)
    elif largest_proportion == 'Gill Sans':
        tk.Label(window,text='Gill Sans',font=('Gill Sans',30)).place(x=950,y=500)
    elif largest_proportion == 'Garamond':
        tk.Label(window, text='Garamond', font=('Garamond', 30)).place(x=950, y=500)

    window.mainloop()


window = tk.Tk()
window.title('face of font')
window.geometry('2400x1100')
canvas1 = tk.Canvas(window,height=9000,width=9000)
photo = tk.PhotoImage(file='./image_file/background.gif')
canvas1.create_image(640,335,anchor='center',image=photo)
canvas1.place(x=1,y=1)

v = tk.StringVar()
v.set(' ')
mission_statement_entry = tk.Entry(window,textvariable=v,font=('Futura',20),width=20).place(x=240,y=320)

v2 = tk.StringVar()
v2.set(' ')
enter_your_idea_font = tk.Entry(window,textvariable=v2,font=('Futura',20),width=20).place(x=240,y=470)


tk.Button(window,text='Get font',background='red',font=('calibri',20),command=recommendations_plus_matching).place(x=300,y=570)




window.mainloop()
