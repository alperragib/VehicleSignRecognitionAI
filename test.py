import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from numpy import array

from keras.models import load_model
model = load_model('traffic_classifier.h5')

classes = { 1:'Hız Limiti (20km/h)',
           2:'Hız Limiti (30km/h)',
           3:'Hız Limiti (50km/h)',
           4:'Hız Limiti (60km/h)',
           5:'Hız Limiti (70km/h)',
           6:'Hız Limiti (80km/h)',
           7:'Hız Limiti Bitti (80km/h)',
           8:'Hız Limiti (100km/h)',
           9:'Hız Limiti (120km/h)',
           10:'Geçiş yok',
           11:'3.5 ton üzerindeki araçlara geçiş yok',
           12:'Kavşakta geçiş hakkı',
           13:'Öncelikli yol',
           14:'Yol ver',
           15:'Dur',
           16:'Araç giremez',
           17:'3.5 ton üzerindeki araçlara yasak',
           18:'Girilmez',
           19:'Dikkat',
           20:'Sola tehlikeli viraj',
           21:'Sağa tehlikeli viraj',
           22:'İki yönlü viraj',
           23:'Engebeli yol',
           24:'Kaygan yol',
           25:'Sağda yol daralıyor',
           26:'Yol çalışması',
           27:'Trafik ışığı',
           28:'Yaya geçidi',
           29:'Çocuk geçidi',
           30:'Bisiklet geçidi',
           31:'Buzlanma olabilir dikkat et',
           32:'Vahşi hayvan geçebilir',
           33:'Geçiş limitleri',
           34:'Sağa dönün',
           35:'Sola dönün',
           36:'Sadece düz gidin',
           37:'Düz veya sağa git',
           38:'Düz veya sola git',
           39:'Sağda kal',
           40:'Solda kal',
           41:'Mecburi döner kavşak',
           42:'Geçiş yasağı bitti',
           43:'3.5 ton üzerindeki araçlara yasak bitti' }

top=tk.Tk()
top.geometry('800x600')
top.title('Trafik İşaretleri Sınıflandırma')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
   global label_packed
   image = Image.open(file_path)
   image = image.resize((30,30))
   image = numpy.expand_dims(image, axis=0)
   image = numpy.array(image)
   predict = model.predict([image])[0]
   en_buyuk = predict[0]
   en_buyuk_indis = 0
   for i in range(len(predict)):
      if en_buyuk<predict[i]:
         en_buyuk=predict[i]
         en_buyuk_indis = i
   sign = classes[en_buyuk_indis+1]
   print(sign)
   label.configure(foreground='#011638', text=sign)
def show_classify_button(file_path):
   print(file_path)
   classify_b=Button(top,text="Test Et",command=lambda: classify(file_path),padx=10,pady=5)
   classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
   classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Görsel Seç",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Trafik İşaretini Kontrol Et",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()