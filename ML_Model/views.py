from django.shortcuts import render
from joblib import load

model = load('./savedModel/model.joblib')

def predictor(request):
    if request.method == 'POST':
        founded_at = request.POST['Founded_at']
        created_at = request.POST['Created_at']
        updated_at = request.POST['Updated_at']
        closed_at = request.POST['Closed_at']
        company_age = request.POST['Company_age']
        investment_rounds = request.POST['Investment_rounds']
        funding_rounds = request.POST['Funding_rounds']
        funding_total_usd = request.POST['Funding_total_usd']
        milestones = request.POST['milestones']
        relationships = request.POST['relationships']
        return_on_investment = request.POST['Return_on_investment']
        
        category_code = request.POST['Category_code']
        if category_code == 'Consulting':
            consulting = 1
            advertising , biotech , ecommerce , games_video , enterprise , software , web , mobile , others = 0,0,0,0,0,0,0,0,0
        elif category_code == 'Advertising':
            advertising = 1
            consulting , biotech , ecommerce , games_video , enterprise , software , web , mobile , others = 0,0,0,0,0,0,0,0,0
        elif category_code == 'Biotech':
            biotech = 1
            advertising , consulting  , ecommerce , games_video , enterprise , software , web , mobile , others = 0,0,0,0,0,0,0,0,0
        elif category_code == 'Ecommerce':
            ecommerce = 1
            advertising , biotech , consulting , games_video , enterprise , software , web , mobile , others = 0,0,0,0,0,0,0,0,0
        elif category_code == 'Games_Video':
            games_video = 1
            advertising , biotech , ecommerce , consulting , enterprise , software , web , mobile , others = 0,0,0,0,0,0,0,0,0
        elif category_code == 'Enterprise':
            enterprise = 1
            advertising , biotech , ecommerce , games_video ,consulting  , software , web , mobile , others = 0,0,0,0,0,0,0,0,0
        elif category_code == 'Software':
            software = 1
            advertising , biotech , ecommerce , games_video , enterprise , consulting , web , mobile , others = 0,0,0,0,0,0,0,0,0
        elif category_code == 'Web':
            web = 1
            advertising , biotech , ecommerce , games_video , enterprise , software ,consulting  , mobile , others = 0,0,0,0,0,0,0,0,0
        elif category_code == 'Mobile':
            mobile = 1
            advertising , biotech , ecommerce , games_video , enterprise , software , web , consulting , others = 0,0,0,0,0,0,0,0,0
        elif category_code == 'Other':
            others  = 1
            advertising , biotech , ecommerce , games_video , enterprise , software , web , mobile , consulting = 0,0,0,0,0,0,0,0,0

        country_code = request.POST['Country_code']
        if country_code == 'AUS':
            aus = 1
            can , esp , deu , fra , gbr , ind , isr , nld , usa , other = 0,0,0,0,0,0,0,0,0,0
        elif country_code == 'CAN':
            can = 1
            aus , esp , deu , fra , gbr , ind , isr , nld , usa , other = 0,0,0,0,0,0,0,0,0,0
        elif country_code == 'ESP':
            esp = 1
            aus , can , deu , fra , gbr , ind , isr , nld , usa , other = 0,0,0,0,0,0,0,0,0,0
        elif country_code == 'DEU':
            deu = 1
            aus , esp , can , fra , gbr , ind , isr , nld , usa , other = 0,0,0,0,0,0,0,0,0,0
        elif country_code == 'FRA':
            fra = 1
            aus , esp , deu , can , gbr , ind , isr , nld , usa , other = 0,0,0,0,0,0,0,0,0,0
        elif country_code == 'GBR':
            gbr = 1
            aus , esp , deu , fra , can , ind , isr , nld , usa , other = 0,0,0,0,0,0,0,0,0,0
        elif country_code == 'IND':
            ind = 1
            aus , esp , deu , fra , gbr , can , isr , nld , usa , other = 0,0,0,0,0,0,0,0,0,0
        elif country_code == 'ISR':
            isr = 1
            aus , esp , deu , fra , gbr , ind , can , nld , usa , other = 0,0,0,0,0,0,0,0,0,0
        elif country_code == 'NLD':
            nld = 1
            aus , esp , deu , fra , gbr , ind , isr , can , usa , other = 0,0,0,0,0,0,0,0,0,0
        elif country_code == 'USA':
            usa = 1
            aus , esp , deu , fra , gbr , ind , isr , nld , can , other = 0,0,0,0,0,0,0,0,0,0
        elif country_code == 'other':
            other = 1
            aus , esp , deu , fra , gbr , ind , isr , nld , usa , can = 0,0,0,0,0,0,0,0,0,0

        y_pred = model.predict([[founded_at, closed_at, investment_rounds,funding_rounds,funding_total_usd, milestones,
         relationships, created_at,updated_at, return_on_investment, company_age, advertising,
         biotech, consulting, ecommerce, enterprise,games_video, mobile, others, software, web,
         aus, can, deu,esp,fra,gbr,ind,isr,nld,usa,other]])
    
        if y_pred[0] == 1:
            y_pred = 'Operating'
        else:
            y_pred = 'Closed'
        return render(request, 'index.html',{'result': y_pred})

    return render(request, 'index.html')
    
