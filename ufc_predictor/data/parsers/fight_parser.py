import bs4
from .base_parser import BaseParser
from tqdm.asyncio import tqdm   

mens_weight_classes = [
    "Strawweight",
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Catch weight",
    "Super lightweight",
    "Welterweight",
    "Super welterweight",
    "Middleweight",
    "Super middleweight",
    "Light heavyweight",
    "Cruiserweight",
    "Heavyweight",
    "Super heavyweight"
    "Open weight"
]

womens_weight_classes = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight"]

class FightParser(BaseParser):
    
    
    async def get_fight_data(self, pbar: tqdm, event_id: str) -> dict:
    
        await self.get_soup()
        # Get fighter IDs and names
        try:
            fighter_ids = self.get_fighter_ids()
            fighter_names = self.get_fighter_names()
        except ValueError as e:
            print(f"Error getting fighter data for {self.url}: {str(e)}")
            return None
        
        # Create mapping of fighter name to ID
        fighter_id_map = dict(zip(fighter_names, fighter_ids))
        
        try:
            fight_details = self.get_fight_details()
        except Exception as e:
            print(f"Error getting fight details for {self.url}: {str(e)}")

            return  {
                'fight_id': self.object_id,
                'error_details': str(e)
            }
        
        # Process each round's stats
        for round_num in range(1, fight_details['round'] + 1):
                # Get both types of stats for this round
            fight_stats = self.get_fight_stats_by_round(round_num)
            significant_strikes = self.get_significant_strikes_by_round(round_num)
                
            # Merge the stats for each fighter
            rounds = []
            for fighter_name in fighter_names:
                rounds.append({
                    'fight_id': self.object_id,
                    'round': round_num,
                    'fighter_id': fighter_id_map[fighter_name],
                    **fight_stats[fighter_name],
                    **significant_strikes[fighter_name]
                })

        fight_details['fighter1_id'] = fighter_ids[0]
        fight_details['fighter2_id'] = fighter_ids[1]

        winner = self.get_fight_winner()

        if winner:
            fight_details['winner_id'] = fighter_id_map[winner]
        else:
            fight_details['winner_id'] = None

        pbar.update(1)
        return {
            'event_id': event_id,
            'fight_id': self.object_id,
            'fighter1_id': fighter_ids[0],
            'fighter2_id': fighter_ids[1],
            'fight_details': fight_details,
            'rounds': rounds
        }
            

    def get_fighter_names(self) -> list[str]:
        fighter_links = self.soup.find_all('a', 'b-link b-fight-details__person-link')

        if not fighter_links:
            raise ValueError(f"Could not find fighter names for {self.url}")
        
        fighter_names = [link.get_text(strip=True) for link in fighter_links]
        return fighter_names
    
    def get_gender(self) -> str:
        gender_element = self.soup.find('i', 'b-fight-details__fight-title')
        if not gender_element:
            raise ValueError(f"Could not find gender for {self.url}")
        
        gender_text = gender_element.get_text(strip=True)
        if "Women's" in gender_text:
            return "F"
        else:
            return "M"
        
    def get_fight_winner(self) -> str:
        # TODO: Handle W/L/NC and decisions
        # TODO: Handle draws?
        # TODO: Handle split decisions?
        for tag in self.soup.find_all('div', class_='b-fight-details__person'):
            for i_text in tag.find_all('i'):
                if "W" in i_text.get_text(strip=True):
                    return tag.find('a', 'b-fight-details__person-link').get_text(strip=True)
        return None
    def get_weight_class(self, gender: str) -> str:
        weight_class_element = self.soup.find('i','b-fight-details__fight-title')
        if not weight_class_element:
            raise ValueError(f"Could not find weight class for {self.url}")
        
        weight_class_text = weight_class_element.text.strip().lower()

        if gender == "F":
            for weight_class in womens_weight_classes:
                if weight_class.lower() in weight_class_text:
                    return weight_class
        elif gender == "M":
            for weight_class in mens_weight_classes:
                if weight_class.lower() in weight_class_text:
                    return weight_class
        else:
            raise ValueError(f"Invalid gender: {gender}")
        
        raise ValueError(f"[{self.url}] Parsed weight class did not match any known weight classes: '{weight_class_text}'")
    
    def get_fight_detail(self, detail: str) -> str:
        detail_element = self.soup.find

    def get_fight_details(self) -> dict:

        try:
            # Get weightclass.
            gender = self.get_gender()

            weight_class = self.get_weight_class(gender)
            
            # Get method
            # Method is unique because the class name for the tag is suffixed with _first
            method_parent_element = self.soup.find('i', 'b-fight-details__text-item_first')
            # Now that we have the parent, fetch the second child i tag
            method_element: bs4.PageElement = method_parent_element.find_all('i')[1]
            method = method_element.get_text(strip=True)

            # Get all fight detail items
            fight_detail_items = self.soup.find_all('i', 'b-fight-details__text-item')

            # Get round
            round_element: bs4.PageElement = fight_detail_items[0]
            round = int(round_element.get_text(strip=True).split(":")[1])
            
            # Get time

            time_element: bs4.PageElement = fight_detail_items[1]
            time_parts = time_element.get_text(strip=True).split(":")
            time = ':'.join(time_parts[1:])

            # # Get format
            format_element: bs4.PageElement = fight_detail_items[2]
            format_parts = format_element.get_text(strip=True).split(":")[1].split(" ")[:-1]
            format = ' '.join(format_parts)

            # # Get referee
            referee_element: bs4.PageElement = fight_detail_items[3]
            referee = referee_element.get_text(strip=True).split(":")[1]
            
            return {
                'fight_id': self.object_id,
                'weight_class': weight_class,
                'gender': gender,
                'method': method,
                'round': round,
                'time': time,
                'format': format,
                'referee': referee
            }
        except Exception as e:
            raise Exception(f"Unable to get fight details for fight: {self.object_id}")

    def get_fight_stats_totals(self) -> dict:
        # Get the second section
        section = self.soup.find_all('section', class_='b-fight-details__section js-fight-section')[1]
        tr = section.find_all('tr')[1]
        return self.parse_fight_stats_tr(tr)

    def get_fight_stats_by_round(self, round: int) -> dict:
        # Get the third section
        section = self.soup.find_all('section', class_='b-fight-details__section js-fight-section')[2]
        tr = section.find_all('tr')[round]
        return self.parse_fight_stats_tr(tr)

    def get_significant_strikes_by_round(self, round: int) -> dict:
        # Get the third section
        section = self.soup.find_all('section', class_='b-fight-details__section js-fight-section')[4]
        tr = section.find_all('tr')[round]

        return self.parse_significant_strikes_tr(tr)

    def _parse_stat_pair(self, element: bs4.PageElement) -> tuple[str, str]:
        """Parse a pair of stats (one for each fighter) from an element"""
        try:
            fighter_elements = element.find_all('p')
            return (
                fighter_elements[0].get_text(strip=True),
                fighter_elements[1].get_text(strip=True)
            )
        except (IndexError, AttributeError):
            print(f"Warning: Could not parse stat pair from element")
            return ('0', '0')

    def _parse_ratio(self, stat: str) -> tuple[int, int]:
        """Parse stats in 'x of y' format, returns (landed, attempted)"""
        try:
            if stat.strip() == '---' or stat.strip() == '--':
                return 0, 0
            parts = stat.split(' of ')
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            print(f"Warning: Could not parse ratio stat: {stat}")
            return 0, 0

    def _parse_percentage(self, stat: str) -> int:
        """Parse percentage stats, returns integer percentage"""
        try:
            if stat.strip() == '---' or stat.strip() == '--':
                return 0
            return int(stat.strip('%'))
        except ValueError:
            print(f"Warning: Could not parse percentage stat: {stat}")
            return 0

    def _parse_time(self, stat: str) -> str:
        """Parse time format stats"""
        return stat.strip()
    
    def get_fighter_ids(self) -> list[str]:
        fighter_links = self.soup.find_all('a', 'b-link b-fight-details__person-link')
        
        if not fighter_links:
            raise ValueError(f"Could not find fighter links for {self.url}")
        
        # Extract fighter IDs from href attributes
        # URL format is like: http://ufcstats.com/fighter-details/[fighter_id]
        fighter_ids = [link['href'].split('/')[-1] for link in fighter_links]
        return fighter_ids

    def parse_fight_stats_tr(self, tr: bs4.PageElement) -> dict:
        try:
            cols = tr.find_all('td')
            fighter_names = self.get_fighter_names()
            fighter_ids = self.get_fighter_ids()
            
            # Create mapping of fighter name to ID
            fighter_id_map = dict(zip(fighter_names, fighter_ids))
            
            # Parse each column
            knockdowns_first, knockdowns_second = self._parse_stat_pair(cols[1])
            sig_str_first, sig_str_second = self._parse_stat_pair(cols[2])
            sig_str_pct_first, sig_str_pct_second = self._parse_stat_pair(cols[3])
            total_str_first, total_str_second = self._parse_stat_pair(cols[4])
            takedowns_first, takedowns_second = self._parse_stat_pair(cols[5])

            def safe_int(val: str) -> int:
                try:
                    if val.strip() == '---' or val.strip() == '--':
                        return 0
                    return int(val.split(' ')[0])
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse integer value: {val}")
                    return 0

            return {
                fighter_names[0]: {
                    'fighter_id': fighter_id_map[fighter_names[0]],
                    'knockdowns': safe_int(knockdowns_first),
                    'sig_str_landed': self._parse_ratio(sig_str_first)[0],
                    'sig_str_attempted': self._parse_ratio(sig_str_first)[1],
                    'sig_str_percent': self._parse_percentage(sig_str_pct_first),
                    'total_str': safe_int(total_str_first),
                    'takedowns': safe_int(takedowns_first)
                },
                fighter_names[1]: {
                    'fighter_id': fighter_id_map[fighter_names[1]],
                    'knockdowns': safe_int(knockdowns_second),
                    'sig_str_landed': self._parse_ratio(sig_str_second)[0],
                    'sig_str_attempted': self._parse_ratio(sig_str_second)[1],
                    'sig_str_percent': self._parse_percentage(sig_str_pct_second),
                    'total_str': safe_int(total_str_second),
                    'takedowns': safe_int(takedowns_second)
                }
            }
        except Exception as e:
            print(f"Error parsing fight stats: {str(e)}")
            # Return empty stats for both fighters
            return {
                fighter_names[0]: {
                    'fighter_id': fighter_id_map[fighter_names[0]],
                    'knockdowns': 0,
                    'sig_str_landed': 0,
                    'sig_str_attempted': 0,
                    'sig_str_percent': 0,
                    'total_str': 0,
                    'takedowns': 0
                },
                fighter_names[1]: {
                    'fighter_id': fighter_id_map[fighter_names[1]],
                    'knockdowns': 0,
                    'sig_str_landed': 0,
                    'sig_str_attempted': 0,
                    'sig_str_percent': 0,
                    'total_str': 0,
                    'takedowns': 0
                }
            }

    def parse_significant_strikes_tr(self, tr: bs4.PageElement) -> dict:
        cols = tr.find_all('td')
        fighter_names = self.get_fighter_names()
        
        # Parse each strike type column
        head_first, head_second = self._parse_stat_pair(cols[3])
        body_first, body_second = self._parse_stat_pair(cols[4])
        leg_first, leg_second = self._parse_stat_pair(cols[5])
        distance_first, distance_second = self._parse_stat_pair(cols[6])
        clinch_first, clinch_second = self._parse_stat_pair(cols[7])

        def parse_fighter_stats(stats_dict):
            result = {}
            for location, stat in stats_dict.items():
                landed, attempted = self._parse_ratio(stat)
                result.update({
                    f'{location}_strike_landed': landed,
                    f'{location}_strike_attempts': attempted,
                    f'{location}_strike_percent': round(landed / attempted if attempted > 0 else 0, 2)
                })
            return result

        first_fighter_stats = parse_fighter_stats({
            'head': head_first,
            'body': body_first,
            'leg': leg_first,
            'distance': distance_first,
            'clinch': clinch_first
        })

        second_fighter_stats = parse_fighter_stats({
            'head': head_second,
            'body': body_second,
            'leg': leg_second,
            'distance': distance_second,
            'clinch': clinch_second
        })

        return {
            fighter_names[0]: first_fighter_stats,
            fighter_names[1]: second_fighter_stats
        }