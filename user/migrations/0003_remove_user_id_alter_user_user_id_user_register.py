# Generated by Django 4.2.5 on 2023-11-12 09:33

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("user", "0002_liveroom"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="user",
            name="id",
        ),
        migrations.AlterField(
            model_name="user",
            name="user_id",
            field=models.CharField(
                max_length=20, primary_key=True, serialize=False, unique=True
            ),
        ),
        migrations.CreateModel(
            name="User_register",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("user_img", models.FileField(upload_to="user_info/")),
                (
                    "user_id",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="user.user"
                    ),
                ),
            ],
        ),
    ]