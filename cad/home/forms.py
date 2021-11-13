from django import forms
from .models import Patient
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy as _


class PatientForm(forms.Form):
    name = forms.CharField(label = 'Name', required = True)
    age = forms.IntegerField(label = 'Age' , required = True)
    sex = forms.ChoiceField(choices =  ((0, "Male"), (1, "Female")),required=True)
    smoking = forms.ChoiceField(choices =  ((0, "Never smoked"), (1, "Ex-Smoker"), (2,'Currently Smoking')),required=True)
    base_week = forms.IntegerField(required = True)
    base_fvc = forms.IntegerField(required = True)
    base_percent = forms.IntegerField(required = True)
    dicoms = forms.FileField(required = True) 
    

'''    def clean_age(self):
        data = self.cleaned_data['age']

        # Check if a date is not in the past.
        if type(data) != type(int):
            raise ValidationError(_('Invalid age'))

        return data

    def clean_sex(self):
        data = self.cleaned_data['sex']

        # Check if a date is not in the past.
        if type(data) != type(int) and data > -1 and data < 2:
            raise ValidationError(_('Invalid sex'))

        return data

    def clean_smoking(self):
        data = self.cleaned_data['smoking']

        # Check if a date is not in the past.
        if type(data) != type(int) and data > -1 and data < 3:
            raise ValidationError(_('Invalid Smoking status'))

        return data

    def clean_fvc(self):
        data = self.cleaned_data['base_fvc']

        # Check if a date is not in the past.
        if type(data) != type(int) and data > -1:
            raise ValidationError(_('Invalid FVC'))

        return data

    def clean_week(self):
        data = self.cleaned_data['base_week']

        # Check if a date is not in the past.
        if type(data) != type(int):
            raise ValidationError(_('Invalid Week'))

        return data

    def clean_percent(self):
        data = self.cleaned_data['base_percent']

        # Check if a date is not in the past.
        if type(data) != type(int) and data > -1 and data < 101:
            raise ValidationError(_('Invalid percentage'))

        return data'''
    
    

    
    