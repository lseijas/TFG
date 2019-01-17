# ********************************************************************************+
# @Author: Laia Seijas
# @Goal: Download multiple search based on queries and a specific amount of images.
# @Date: 10/12/2018
# *********************************************************************************
import re, os, sys, datetime, time
import pandas
from selenium import webdriver
from contextlib import closing
from selenium.webdriver import Firefox
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from pattern.web import URL, extension, cache, plaintext, Newsfeed, DOM
import urllib.request as urllib2
from urllib.parse import unquote
import csv

# ********************************************************************************+
# @Description: Download Images from google chrome specifying an image size. 
# @Additional information: Maximum amount of images per query, around 250.
# *********************************************************************************
class GoogleImageExtractor(object):

    #Initialization of the GoogleSearcher.
    def __init__(self, queries_list = " "):

        #queries_list is a list which contains all the queries that you want to search for.
        self.queries_list = queries_list
        #Number of images that you would like to download for each search. 
        self.num_images = 900

        #Specific query that you are currently searching for.
        self.query_search = " "

        #URL definition
        self.prefix_of_search_url = "https://www.google.com.sg/search?q="
        self.postfix_of_search_url = '&source=lnms&tbm=isch&sa=X&ei=0eZEVbj3IJG5uATalICQAQ&ved=0CAcQ_AUoAQ&biw=939&bih=591'
        self.url_search = '' #What to search for.

        #Store the url from each image.
        self.images_url_list = []

        #Directory path where you want to store the images.
        self.directory_path = "/Users/lseijas/Desktop/TFG_Code/Image"

    #Specification of the directory path where you want to store the images.
    def setDirectoryPath(self, directory_name):
        self.directory_path = directory_name

    #Specification of the number of images that you want to retrieve from each query.
    def setNumberOfImages(self,num_images):
        self.num_images = num_images

    #To transform the spaces from the query that you want to search from into a "+", to be able to search from it.
    def reformat_search_for_spaces(self):
        self.query_search = self.query_search.rstrip().replace(' ','+')

    #Generate the complete url using the prefix, the query that you want to search and the postfix.
    def CreateSearchUrl(self):
        #Create a proper query to be able to search.
        self.reformat_search_for_spaces()
        #Update the URL text.
        self.url_search = self.prefix_of_search_url + self.query_search + self.postfix_of_search_url

    #Exctract the url from the different images using Selenium and Google Chrome (chromedriver)
    def extract_images_url(self):

        #Initialize Chrome Webdriver using Selenium.
        driver = webdriver.Chrome("/usr/local/bin/chromedriver")
        driver.get(self.url_search)

        #Scroll around google page.
        init_position = 0
        move_to = 200000
        not_find = False
        for scroll in range(30):
            window_scroll = "window.scrollBy("+str(init_position)+","+str(move_to)+")"
            driver.execute_script(window_scroll)
            time.sleep(0.2)
            init_position = move_to
            move_to = move_to + 100000
            #Find the "show more results button"
            try:
                #Click the "Show more results"
                driver.find_element_by_xpath("//input[@type='button']").click()
                print("Click!")
            except:
                continue
        time.sleep(0.5)
        self.driver_source = driver.page_source

        #Retrieve the different images-url from the google page. 
        dom = DOM(self.driver_source)
        tag_list = dom('a.rg_l')
        print("Total images retrieved: " + str(len(tag_list)))
        #Avoid trying to retrieve more images that the ones that google allows.
        if (self.num_images > len(tag_list)):
            self.num_images = len(tag_list)
        #Only allow a maximum number of images defined by, self.num_images.
        for tag in tag_list[:self.num_images]:
            tar_str = re.search('imgurl=(.*)&imgrefurl', tag.attributes['href'])
            try:
                self.images_url_list.append(tar_str.group(1))
            except:
                print ('error parsing', tag)

        #Print number of images that you were able to download
        #(always a little bit less than self.num_images, since Exceptions and permission problems to some websites.)
        print("\nTotal number of URL images: " + str(len(self.images_url_list)))
        #Close the Google Chrome Webdriver.
        driver.quit()
    
    #Downloading images to the specific folder directory.
    #Each query will have a particular sub-folder containing the images. 
    def downloading_all_photos(self):
        #Create the general directory path in case that do not exist:
        if not os.path.exists(self.directory_path):
            os.mkdir(self.directory_path)

        #Create the sub-directory (one per query) in case that do not exist:
        query =  self.query_search.split()[0]
        DIR = os.path.join(self.directory_path, "Dataset")
        if not os.path.exists(DIR):
            os.mkdir(DIR)

        #Go to all the URL images and download the images in the specific folder. 
        i = 1
        for url_link in self.images_url_list:
            #Need to decode the url.
            url = unquote(url_link)
            #Extract the type from the image url using the urllib library.
            try:
                response = urllib2.urlopen(url)
                info = response.info()

                #Verify that the image type is a proper one ( we only want .jpg, .png and .jpeg images)
                if (info.get_content_subtype() == "jpg" or info.get_content_subtype() == "png" or info.get_content_subtype() == "jpeg"):
                    #Name of the image.
                    image_path = query + "_image_" + str(i) + "." + str(info.get_content_subtype())
                    directory_path = os.path.join(DIR,image_path)
                    try:
                        #Retrieve the image and increment the counter.
                        raw_img = urllib2.urlretrieve(str(url), directory_path)
                        print("Storing image number " + str(i) + " from the url: " + str(url))
                        #We put image attributes in the CSV file
                        try:
                            #We open again the CSV file to add a row for each image
                            with open('data.csv', 'a') as csvfile:
                                csvwriter = csv.writer(csvfile)
                                #We define the row depending on the query we are looking at
                                if query == 'wildfire' or query == 'incendios+forestales' or query == 'incêndios' or query == 'forest+fire' or query == 'california+fire':
                                    row = [query + "_image_" + str(i), 1, 0, 0, 0]
                                if query == 'building+fire' or query == 'edificios+llamas' or query == 'Feuer+Gebäude' or query == 'edifici+incendi' or query == 'edificios+historicos+llamas':
                                    row = [query + "_image_" + str(i), 0, 1, 0, 0]
                                if query == 'forest' or query == 'nature' or query == 'bosc' or query == 'bosque' or query == 'la+mola':
                                    row = [query + "_image_" + str(i), 0, 0, 1, 0]
                                if query == 'building' or query == 'edificios' or query == 'edifici' or query == 'immeubles' or query == 'edificios+historicos':
                                    row = [query + "_image_" + str(i), 0, 0, 0, 1]
                                csvwriter.writerow(row)
                        except Exception as e:
                            print ("Exception al escriure al fitxer ")
                        i = i + 1
                    except Exception as e:
                        print ("Exception in image number : " +str(i)+ str(e))
            except Exception as e:
                print("Exception")


    #Searching from multiple queries (using queries_list)
    def multi_search_download(self):
        #Go through all the queries that you want to retrieve.
        for query in self.queries_list:
            #Initialize the lists from the images.
            self.images_url_list = []
            self.images_info_list = []
            #Specific query that you want to search for.
            self.query_search = query
            #Search and download images from the query.
            self.CreateSearchUrl()
            self.extract_images_url()
            self.downloading_all_photos()

#********************
#<div id="smbw"> <input class="ksb" value="Més resultats" id="smb" data-lt="S'està carregant..." type="button" data-ved="0ahUKEwjcy566_JffAhUuhqYKHSvQClkQxdoBCFc"> </div> 
#********************

if __name__ == '__main__':
    #Make a CSV file that includes the name of the image an other properties 
    """fields = ['Name', 'FireForest', 'FireCity', 'Forest', 'City']
    filename = 'data.csv'
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)  """
    #Queries that you want to search for:
    #queries_list = ["wildfire","building+fire","forest","nature","bosc","bosque","building", "edificios","edifici", "immeubles","incendios+forestales",
    #queries_list = ["incêndios","forest+fire","edificios+llamas","Feuer+Gebäude","edifici+incendi"]
    queries_list = ["california+fire", "la+mola", "edificios+historicos", "edificios+historicos+llamas"]
    #Instantiation of the google class.
    w = GoogleImageExtractor(queries_list)
    #Download specific amount of images per query (max 250 - at this stage)
    w.setNumberOfImages(200)
    #Specification of the directory path where you want to download the images.
    #w.setDirectoryPath("/TOSHIBALAIA⁩/⁨Dataset/⁩")
    w.setDirectoryPath("/Users/lseijas/Desktop/TFG_Code/Image")
    #Search for each query.
    w.multi_search_download()
    w.multi_search_download()
