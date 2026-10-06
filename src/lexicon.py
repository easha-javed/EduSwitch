"""Small word lists for the dictionary LID baseline and the language type heuristic.

These are intentionally small. Extend them as you read the data (Rida and Hareem own this file).
Optionally drop a bigger English list at data/raw/english_words.txt (one word per line).
"""

UR_WORDS = set("""
hai hain hy hay hn ho hoga hogi hoon hun tha thi hua hui hue hota hoti hote
ka ki ke ko se mein mai par pe tak ne ab
kya kia kiya kab kb kahan kahaan kaise kese kyun kyu kitna kitni kitne kon kaun konsa
mera meri mere mra mri mjhe mujhe mujh mujhy hamara hamari hum aap ap tum tumhara
nahi nahin nhi nai na
karna karni karne krna krni krne kar kr karo karen karta karti karte raha rahi rahe rha rhi
chahiye chahta chahti chahtay chahte
agar lekin magar aur ya bhi hi toh phir abhi jab tab
ye yeh wo woh iss uss yahan wahan
bata batao batain bataye bataein
dena deni dene diya liya lena leni
sakta sakti sakte jaye jaega jayegi
kuch sab bohat bahut zyada kam
din raat kal aaj parso
mil mila mili milega milegi milay
lagta lagti
gaya gayi gaye
hona hone
challan jama
""".split())

EN_WORDS = set("""
a an the and or but if so because before after during while of in on at to for with from by about into over under
is are was were be been am i you he she it we they me my your our their this that these those there here
can could will would should may might must do does did have has had not no yes please thanks thank
sir madam teacher professor instructor
exam exams assignment assignments quiz midterm final finals paper course courses class classes lecture lab labs
semester fee fees dues payment grade grades marks result results gpa cgpa attendance absent present
registration register withdraw withdrawal drop add credit credits section schedule timetable time date deadline
extend extension submit submission upload download login password account portal lms email link
monday tuesday wednesday thursday friday saturday sunday today tomorrow yesterday week month year
january february march april may june july august september october november december
what when where why how which who whom whose
need want know help issue problem error working work able still again later early late
exam hall room campus university department office admin
confirm check update change changed cancel cancelled postponed
get got give take make send sent open close
""".split())
