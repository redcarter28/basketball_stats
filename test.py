import basketball_reference_scraper.players as brs

print(brs.get_stats('Tyrese Haliburton', stat_type='PER_GAME', playoffs=True, career=False))