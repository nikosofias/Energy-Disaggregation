Ο φάκελος περιέχει 3 αρχεία με σύνολα δεδομένων, το Total_house__Water_heater.csv, το fridge.csv και το Hotplate.csv για το διάστημα 2-3 Ιουνίου 2022. 
Η συχνότητα δειγματοληψίας είναι 1sample/1minute, ενώ όταν εσωτερικά στο μετρητή εντοπίζεται κάποι αλλαγή στο μέγεθος της ισχύος, συγκεκιρμένα
όταν ανιχνευθεί αλλαγή +-100Watt, τότε ο μετρητής στέλνει τα δεδομένα με συχνότητα 1sample/50millisecond για 3 δευτερόλεπτα και μετά 
επανέρχεται στην κανονική του συχνότητα.
Αυτό συμβαίνει προκειμένου να έχουμε σε υψηλή ανάλυση τις χρονικές στιγμές κατά τις οποίες έχουμε είσοδο/έξοδο ενός νέου φορτίου.
Κατά το διάστημα της καταγραφής συναντάμε ενεργοποιήσεις των αντιστασιακών (resistive) φορτίων του θερμοσίφωνα και ενός ματιού κουζίνας 
(αυτόνομη συσκευή που μπαίνει στην πρίζα), γι αυτό και αποτελεί καλό παράδειγμα για τις αρχικές παρατηρήσεις.

Στο αρχείο Total_house__Water_heater.csv βρίσκουμε τις παρακάτω στήλες:
1) Date -->Ημερομηνία σε μορφή YYYY-MM-DD
2) Time (UTC) --> Ώρα σε UTC με ακρίβεια Millisecond
3) Active Power L1 (W) --> Ενεργός ισχύς συνόλου του σπιτιού σε Watt
4) Reactive Power L1 (Var) --> Άεργος ισχύς συνόλου του σπιτιού σε Var
5) Active Power L3 (W) --> Ενεργός ισχύς του θερμοσίφωνα σε Watt

Αντίστοιχα τα υπόλοιπα αρχεία περιέχουν τις καταγραφές για τις υπόλοιπες συσκευές, για το ίδιο χρονικό διάστημα:
1) Date -->Ημερομηνία σε μορφή YYYY-MM-DD
2) Time (UTC) --> Ώρα σε UTC με ακρίβεια Millisecond
3) Active Power L1 (W) --> Ενεργός ισχύς εστίας μαγειρέματος σε Watt
 

