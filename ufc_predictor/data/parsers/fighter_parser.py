import bs4
import re
from datetime import datetime
from ufc_predictor.data.parsers.base_parser import BaseParser
from tqdm.asyncio import tqdm   

class FighterParser(BaseParser):
    
    async def get_fighter_data(self, pbar: tqdm) -> dict:

        try:
            await self.get_soup()

            attribute_ul = self.soup.find('ul', class_='b-list__box-list')

            
            data =  {
                'fighter_id': self.object_id,
                'name': self.get_fighter_name(),
                'nickname': self.get_fighter_nickname(),
                'height_feet': self.get_fighter_height(attribute_ul)[0],
                'height_inches': self.get_fighter_height(attribute_ul)[1],
                'weight': self.get_fighter_weight(),    
                'reach': self.get_fighter_reach(),
                'stance': self.get_fighter_stance(),
                'dob': self.get_fighter_dob()
            }
            pbar.update(1)
            return data
        except Exception as e:
            return {
                'name': None,
                'nickname': None,
                'height_feet': None,
                'height_inches': None,
                'weight': None,    
                'reach': None,
                'stance': None,
                'dob': None
            }
    
    def get_fighter_dob(self) -> str:

        try:
            attribute_li = self.soup.find_all('li', class_="b-list__box-list-item b-list__box-list-item_type_block")
            dob_text = attribute_li[4].get_text(strip=True).split(':')[1]
            dob = datetime.strptime(dob_text, '%b %d, %Y')
        except Exception:
            return None
        return dob
    
    def get_fighter_name(self) -> str:
        try:
            name_elem = self.soup.find('span', class_='b-content__title-highlight')
            if name_elem is None:
                raise ValueError(f"Fighter name not found for fighter_id: {self.fighter_id}")
        except Exception:
            return None
        return name_elem.get_text(strip=True)

    def get_fighter_nickname(self) -> str:
        try:
            nickname_elem = self.soup.find('p', class_='b-content__Nickname')
        except Exception:
            return None
        return nickname_elem.get_text(strip=True)


    def get_fighter_height(self, attribute_ul: bs4.PageElement) -> tuple[int, int]:

        def parse_feet_inches(text):
            match = re.match(r'(\d+)\'\s*(\d+)\"', text)
            if match:
                feet = int(match.group(1))
                inches = int(match.group(2))
                return feet, inches
            else:
                return None
        
        attribute_li = attribute_ul.find_all('li', class_="b-list__box-list-item b-list__box-list-item_type_block")
        height_text = attribute_li[0].get_text(strip=True).split(':')[1]

        return parse_feet_inches(height_text)

    def get_fighter_weight(self) -> str:
        try:
            attribute_li = self.soup.find_all('li', class_="b-list__box-list-item b-list__box-list-item_type_block")
            weight_text = attribute_li[1].get_text(strip=True).split(':')[1].split(' ')[0]
            weight = int(weight_text)
        except Exception:
            return None
        return weight

    def get_fighter_reach(self) -> str:
        def extract_digits(text):
            match = re.search(r'\d+', text)  # \d+ matches one or more digits
            if match:
                return match.group(0)
            else:
                return None
        attribute_li = self.soup.find_all('li', class_="b-list__box-list-item b-list__box-list-item_type_block")
        reach_text = attribute_li[2].get_text(strip=True)
        reach = extract_digits(reach_text)
        return reach

    def get_fighter_stance(self) -> str:
        try:
            attribute_li = self.soup.find_all('li', class_="b-list__box-list-item b-list__box-list-item_type_block")
        
            stance_text = attribute_li[3].get_text(strip=True).split(':')[1]
        except Exception:
            return None
        return stance_text