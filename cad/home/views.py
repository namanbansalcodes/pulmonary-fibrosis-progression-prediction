from django.shortcuts import render
from django.http import HttpResponse
from .models import Patient
from .dash import Dash
from .forms import PatientForm
import pickle
import pydicom


# Create your views here.
def index(request):
    context = {
        'title': 'Home'
    }
    return render(request, 'homepage.html', context)


def form(request):
    if request.method == 'POST':
        pform = PatientForm(request.POST, request.FILES)

        if pform.is_valid():
            p = Patient(
                name = request.POST['name'], 
                age = request.POST['age'], 
                sex = request.POST['sex'], 
                smoking = request.POST['smoking'], 
                base_week = request.POST['base_week'], 
                base_percent = request.POST['base_percent'], 
                dicoms = request.FILES['dicoms']
            )
            
            p.save()
            
            dashapp = Dash(p)
        
            return render(request, 'results.html', {'title':'Results'})
        

    context = {
        'title': 'Predict', 
        'form': PatientForm(), 
    }

    return render(request, 'form.html', context)

def handle_uploaded_file(details, files):
    lungdata = []
    for f in files:
        lungdata.append(f)

    with open(f"static/lungdata{details['name'].split()[0]}.pickle", "wb") as file:
        pickle.dump(lungdata, file) 
