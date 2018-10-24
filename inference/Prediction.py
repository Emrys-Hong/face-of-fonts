import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

data = pd.read_csv('vector_recommender.csv')

font_form = pd.DataFrame(data,)

class Vector_composition():


    def __init__(self,helvetica_neue, helvetica, arial,myraid,futura,gotham,frutiger,gillsans,garamond):
        self.helvetica_neue = helvetica_neue
        self.helvetica = helvetica
        self.arial = arial
        self.myraid = myraid
        self.futura = futura
        self.gotham = gotham
        self.frutiger = frutiger
        self.gillsans = gillsans
        self.garamond = garamond




    def vector_composition(self):

        composition_list = []
        for index in range(200):
            composition_list.append(font_form.ix[index,"Roboto regular"]*self.helvetica_neue +font_form.ix[index,"Doppio One regular"]*self.helvetica+
                                font_form.ix[index,"Shanti regular"]*self.arial+font_form.ix[index,"PT Sans regular"]*self.myraid+font_form.ix[index,"Nunito regular"]*self.futura
                                +font_form.ix[index,"Finger Paint regular"]*self.gotham+font_form.ix[index,"Istok Web regular"]*self.frutiger+
                                font_form.ix[index,"Oxygen regular"]*self.gillsans+font_form.ix[index,"EB Garamond regular"]*self.garamond)

        font_vector = pd.DataFrame(np.reshape(composition_list,(200,1)),columns=["composition_vector"])
        self.font_form = pd.concat([font_form,font_vector],axis=1,join_axes=[font_form.index])




        



    def found_closest_font_euclidean(self):

        distance_dict = {}


        for columns in self.font_form:
            count = 0
            for index in range(200):
                count += (float(self.font_form.ix[index,"composition_vector"]) - float(self.font_form.ix[index, columns])) ** 2

            distance_dict[columns] = count

        tuples = sorted(distance_dict.items(), key=lambda x: x[1])

        return tuples

    def found_closest_font_cosine(self):
        self.cosine_dict = {}
        for columns in self.font_form:
            self.cosine_dict[columns] = cosine(self.font_form.ix[:,'composition_vector'], self.font_form.ix[:,columns])
        tuple = sorted(self.cosine_dict.items(), key=lambda x: x[1])

        return tuple

    def find_cosine_similarity(self,name):
        value = cosine(self.font_form.ix[:,'composition_vector'], self.font_form.ix[:,name])
        return value



#font_vector = Vector_composition( 0.15624192, 0.056223955,  0.013977319, 0.02745132,  0.59628558, 0.0142462,  0.10929669,0.020677818, 0.0055992017)
#font_vector.vector_composition()
#print (font_vector.found_closest_font_euclidean())
#print (font_vector.found_closest_font_cosine())


'''
def vector_composition(helvetica_neue, helvetica, arial,myraid,futura,gotham,frutiger,gillsans,garamond):

    composition_list = []
    for index in range(200):
        composition_list.append(font_form.ix[index,"Roboto regular"]*helvetica_neue+font_form.ix[index,"Doppio One regular"]*helvetica+
                                font_form.ix[index,"Shanti regular"]*arial+font_form.ix[index,"PT Sans regular"]*myraid+font_form.ix[index,"Nunito regular"]*futura
                                +font_form.ix[index,"Finger Paint regular"]*gotham+font_form.ix[index,"Istok Web regular"]*frutiger+
                                font_form.ix[index,"Oxygen regular"]*gillsans+font_form.ix[index,"EB Garamond regular"]*garamond)

    font_vector = pd.DataFrame(np.reshape(composition_list,(200,1)),columns=["composition_vector"])
    

    new_font_form = pd.concat([font_form,font_vector],axis=1,join_axes=[font_form.index])



    return new_font_form

new_font_form = vector_composition(1,2,3,4,5,6,7,8,9)
print (new_font_form)

def found_closest_font(name):

    distance_dict = {}

    for columns in font_form:
        count = 0
        for index in range(200):
            count += (float(new_font_form.ix[index,name])-float(new_font_form.ix[index,columns]))**2

        distance_dict[columns] = count

    tuple = sorted(distance_dict.items(), key=lambda x: x[1])

    return tuple




print (found_closest_font('composition_vector'))
'''
