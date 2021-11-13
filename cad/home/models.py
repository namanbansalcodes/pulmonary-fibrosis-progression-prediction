from django.db import models
import uuid

# Create your models here.


class Patient(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length = 30, default= 'Name')
    age = models.IntegerField(default=50)
    sex = models.IntegerField(default=1)
    smoking = models.IntegerField(default=3)
    base_week = models.IntegerField(default=0)
    base_fvc = models.IntegerField(default=2000)
    base_percent = models.IntegerField(default=50)
    dicoms = models.FileField(upload_to='./static', default='static/anonymous.zip')
