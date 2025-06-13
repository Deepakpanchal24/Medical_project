import datetime

def calculate_age(birth_date):
    """Calculate age in years based on birth date."""
    today = datetime.date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age