from functools import reduce
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

COUNTRIES = ['rwanda', 'djibouti', 'kenya', 'seychelles', 'uganda', 'tanzania', 'mayotte', 'réunion', 'zambia', 'madagascar', 'eritrea', 'somalia', 'ethiopia', 'burundi', 'zimbabwe', 'mauritius', 'malawi', 'british_indian_ocean_territory', 'comoros', 'mozambique', 'são_tomé_and_príncipe', 'angola', 'equatorial_guinea', 'dr_congo', 'chad', 'cameroon', 'central_african_republic', 'gabon', 'republic_of_the_congo', 'south_sudan', 'egypt', 'western_sahara', 'morocco', 'libya', 'tunisia', 'algeria', 'sudan', 'swaziland', 'south_africa', 'namibia', 'botswana', 'lesotho', 'gambia', 'sierra_leone', 'benin', 'mauritania', 'liberia', 'togo', 'cape_verde', 'burkina_faso', 'senegal', 'ivory_coast', 'guinea', 'ghana', 'guinea-bissau', 'mali', 'nigeria', 'niger', 'turks_and_caicos_islands', 'jamaica', 'sint_maarten', 'martinique', 'united_states_virgin_islands', 'cuba', 'curaçao', 'bahamas', 'dominican_republic', 'aruba', 'montserrat', 'dominica', 'haiti', 'trinidad_and_tobago', 'anguilla', 'saint_kitts_and_nevis', 'saint_lucia', 'barbados', 'puerto_rico', 'guadeloupe', 'cayman_islands', 'saint_barthélemy', 'grenada', 'saint_vincent_and_the_grenadines', 'antigua_and_barbuda', 'saint_martin', 'british_virgin_islands', 'panama', 'nicaragua', 'honduras', 'costa_rica', 'el_salvador', 'guatemala', 'belize', 'canada', 'united_states', 'saint_pierre_and_miquelon', 'mexico', 'bermuda', 'united_states_minor_outlying_islands', 'greenland', 'peru', 'bolivia', 'chile', 'french_guiana', 'suriname', 'falkland_islands', 'guyana', 'ecuador', 'brazil', 'uruguay', 'south_georgia', 'colombia', 'argentina', 'paraguay', 'venezuela', 'kazakhstan', 'turkmenistan', 'uzbekistan', 'kyrgyzstan', 'tajikistan', 'macau', 'hong_kong', 'taiwan', 'north_korea', 'japan', 'mongolia', 'south_korea', 'china', 'philippines', 'myanmar', 'indonesia', 'laos', 'brunei', 'thailand', 'timor-leste', 'cambodia', 'singapore', 'vietnam', 'malaysia', 'bangladesh', 'afghanistan', 'maldives', 'sri_lanka', 'nepal', 'india', 'pakistan', 'iran', 'bhutan', 'qatar', 'iraq', 'azerbaijan', 'oman', 'yemen', 'bahrain', 'kuwait', 'israel', 'lebanon', 'turkey', 'syria', 'saudi_arabia', 'jordan', 'armenia', 'united_arab_emirates', 'georgia', 'palestine', 'slovakia', 'czechia', 'poland', 'romania', 'russia', 'hungary', 'cyprus', 'kosovo', 'bulgaria', 'ukraine', 'belarus', 'moldova', 'denmark', 'norway', 'ireland', 'finland', 'isle_of_man', 'åland_islands', 'svalbard_and_jan_mayen', 'united_kingdom', 'faroe_islands', 'estonia', 'jersey', 'iceland', 'guernsey', 'latvia', 'sweden', 'lithuania', 'serbia', 'vatican_city', 'montenegro', 'albania', 'andorra', 'italy', 'greece', 'spain', 'san_marino', 'gibraltar', 'bosnia_and_herzegovina', 'macedonia', 'malta', 'slovenia', 'portugal', 'croatia', 'switzerland', 'liechtenstein', 'monaco', 'germany', 'netherlands', 'luxembourg', 'belgium', 'austria', 'france', 'australia', 'norfolk_island', 'new_zealand', 'christmas_island', 'cocos_keeling_islands', 'new_caledonia', 'vanuatu', 'solomon_islands', 'fiji', 'papua_new_guinea', 'marshall_islands', 'kiribati', 'palau', 'micronesia', 'northern_mariana_islands', 'guam', 'nauru', 'samoa', 'niue', 'tonga', 'pitcairn_islands', 'french_polynesia', 'wallis_and_futuna', 'american_samoa', 'cook_islands', 'tokelau', 'tuvalu']
SUBREGIONS = ['eastern_africa', 'middle_africa', 'northern_africa','southern_africa', 'western_africa' , 'caribbean', 'central_america', 'northern_america', 'south_america', 'central_asia', 'eastern_asia', 'south-eastern_asia', 'southern_asia', 'western_asia','central_europe', 'eastern_europe', 'northern_europe', 'southern_europe', 'western_europe', 'australia_and_new_zealand', 'melanesia', 'micronesia', 'polynesia']
REGIONS = ['africa', 'americas', 'asia', 'europe', 'oceania']

def visualise_matrix(matrix, names):
    idxs = [list(names).index(c) for c in filter(lambda x: x in names, COUNTRIES)]
    matrix = matrix[idxs][:, idxs]
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)
    return fig

def get_located_in(kg):
	subregion_dict = {}
	region_dict = {}

	for subregion in SUBREGIONS:
		subregion_dict[subregion] = {y[0] for y in filter(lambda x: x[2] == subregion, kg)}

	for region in REGIONS:
		region_dict[region] = {y[0] for y in filter(lambda x: x[2] == region, kg)}

	return region_dict, subregion_dict

def get_full_kg():
	base_path = Path(__file__).parent / 'data'
	with open(base_path / 'raw' / 'countries_S0.tsv', 'r') as f:
		full_kg = [tuple(line.strip().split('\t')) for line in f.readlines()]

	with open(base_path / 'raw' / 'val.tsv', 'r') as f:
		full_kg += [tuple(line.strip().split('\t')) for line in f.readlines()]

	with open(base_path / 'raw' / 'test.tsv', 'r') as f:
		full_kg += [tuple(line.strip().split('\t')) for line in f.readlines()]

	return get_located_in(full_kg)

def get_task_kg(task_name):
	base_path = Path(__file__).parent / 'data'
	with open(base_path / 'raw' / f'countries_{task_name}.tsv', 'r') as f:
		task_kg = [tuple(line.strip().split('\t')) for line in f.readlines()]

	return get_located_in(task_kg)

def make_region_matrices():
	region_matrix = np.zeros((len(COUNTRIES), len(COUNTRIES)))
	subregion_matrix = np.zeros((len(COUNTRIES), len(COUNTRIES)))

	all_regions, all_subregions = get_full_kg()

	all_regions_closed = all_regions.copy()
	for region in all_regions:
		for subregion in SUBREGIONS:
			if subregion in all_regions[region]:
				all_regions_closed[region].update(all_subregions[subregion])

	offset = 0
	for region in all_regions:
		all_regions_filtered = list(filter(lambda x: x in COUNTRIES, all_regions_closed[region]))
		length = len(all_regions_filtered)
		region_matrix[offset:offset + length, offset:offset + length] = 1
		offset += length

	offset = 0
	for subregion in all_subregions:
		length = len(all_subregions[subregion])
		subregion_matrix[offset:offset + length, offset:offset + length] = 1
		offset += length

	return region_matrix, subregion_matrix

def get_located_in_matrix(task_name):
	region_dict, subregion_dict = get_task_kg(task_name)

	region_matrix, subregion_matrix = make_region_matrices()

	for country in COUNTRIES:
		if not any(country in region_dict[region] for region in region_dict):
			region_matrix[COUNTRIES.index(country),:] = 0
			region_matrix[:,COUNTRIES.index(country)] = 0
		if not any(country in subregion_dict[subregion] for subregion in subregion_dict):
			subregion_matrix[COUNTRIES.index(country),:] = 0
			subregion_matrix[:,COUNTRIES.index(country)] = 0

	return region_matrix, subregion_matrix

if __name__ == "__main__":
	TASK = "S3"

	region_matrix, subregion_matrix = get_located_in_matrix(TASK)

	fig1 = visualise_matrix(region_matrix, COUNTRIES)
	fig2 = visualise_matrix(subregion_matrix, COUNTRIES)
	fig1.suptitle(f"Region Matrix {TASK}")
	fig2.suptitle(f"Subregion Matrix {TASK}")

	plt.show()

