import netron
import Dashboard
from multiprocessing import Process

# Attenzione, se si modifica la porta di netron, e' necessario aggiornare il dato anche nell'iframe in Dashboard.py
netron.start('model/model.h5', log=False, browse=False)
proc = Process(target=Dashboard.app.run_server(port=8082))
