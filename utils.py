import re 
# for validating an email 
def validate_email(email): 
    regex = '^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$'

    if(re.search(regex,email)): 
            return True        
    else: 
            return False