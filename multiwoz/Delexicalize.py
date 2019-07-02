import re
import os
# import simplejson as json
import json

digitpat = re.compile('\d+')
timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")

fin = open('data/multi-woz/mapping.pair', 'r')
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

# FORMAT
# domain_value
# restaurant_postcode
# restaurant_address
# taxi_car8
# taxi_number
# train_id etc..

def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    text = re.sub(timepat, ' [value_time] ', text)
    text = re.sub(pricepat, ' [value_price] ', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def prepareSlotValuesIndependent():
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    dic = []
    dic_area = []
    dic_food = []
    dic_price = []
    # dic_postcoce=[]

    # temp=dict()

    # read databases
    for domain in domains:
        try:
            fin = open('db/' + domain + '_db.json')
            db_json = json.load(fin)
            fin.close()

            for ent in db_json:
                for key, val in ent.items():
                    if type(val) != str:
                        continue

                    # if normalize(val) in temp and domain != temp[normalize(val)]:
                    #     if 'value' not in normalize(val):
                    #         print(key, val, normalize(val), domain, temp[normalize(val)])
                    # else:
                    #     temp[normalize(val)]=domain

                    if val == '?' or val == 'free':
                        pass
                    elif key == 'address':
                        dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        if "road" in val:
                            val = val.replace("road", "rd")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "rd" in val:
                            val = val.replace("rd", "road")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "st" in val:
                            val = val.replace("st", "street")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "street" in val:
                            val = val.replace("street", "st")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif key == 'name':
                        dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        if "b & b" in val:
                            val = val.replace("b & b", "bed and breakfast")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "bed and breakfast" in val:
                            val = val.replace("bed and breakfast", "b & b")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "hotel" in val and 'gonville' not in val:
                            val = val.replace("hotel", "")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "restaurant" in val:
                            val = val.replace("restaurant", "")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif key == 'postcode':
                        dic.append((normalize(val), '[' + domain + '_' + 'postcode' + ']'))
                    elif key == 'phone':
                        dic.append((val, '[' + domain + '_' + 'phone' + ']'))
                    elif key == 'trainID':
                        dic.append((normalize(val), '[' + domain + '_' + 'id' + ']'))
                    elif key == 'department':
                        dic.append((normalize(val), '[' + domain + '_' + 'department' + ']'))

                    # NORMAL DELEX
                    elif key == 'area':
                        dic_area.append((normalize(val), '[' + 'value' + '_' + 'area' + ']'))
                    elif key == 'food':
                        dic_food.append((normalize(val), '[' + 'value' + '_' + 'food' + ']'))
                    elif key == 'pricerange':
                        dic_price.append((normalize(val), '[' + 'value' + '_' + 'pricerange' + ']'))
                    # elif key == 'postcode':
                    #     dic_postcoce.append((normalize(val), '[' + 'value' + '_' + 'postcode' + ']'))
                    else:
                        pass
                    # TODO car type?
        except:
            print('error: ', domain)
            pass

        if domain == 'hospital':
            dic.append((normalize('Hills Rd'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('Hills Road'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('CB20QQ'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('0122324515', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize('Addenbrookes Hospital'), '[' + domain + '_' + 'name' + ']'))

        elif domain == 'police':
            dic.append((normalize('Parkside'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('CB11JG'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize('Parkside Police Station'), '[' + domain + '_' + 'name' + ']'))

    # add at the end places from trains
    fin = open('db/' + 'train' + '_db.json')
    db_json = json.load(fin)
    fin.close()

    for ent in db_json:
        for key, val in ent.items():
            if key == 'departure' or key == 'destination':
                dic.append((normalize(val), '[' + 'value' + '_' + 'place' + ']'))

    # add specific values:
    for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        dic.append((normalize(key), '[' + 'value' + '_' + 'day' + ']'))

    # more general values add at the end
    dic.extend(dic_area)
    dic.extend(dic_food)
    dic.extend(dic_price)
    # dic.extend(dic_postcoce)

    ret_dic=dict()
    for key, value in dic:
        if key in ret_dic:
            ret_dic[key].add(value)
        else:
            ret_dic[key]=set([value])

    return ret_dic


def delexicalise(utt, dictionary, domains):
    for key in dictionary:
        vals=dictionary[key]
        if len(vals)>1:
            for val in vals:
                d_r=val.split('_')
                d=d_r[0][1:]
                r=d_r[1][:-1]
                if d in domains and 'reqt' in domains[d] and r in domains[d]['reqt']:
                    utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
                    utt=utt.strip()
        else:
            utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + list(vals)[0] + ' ')
            utt = utt.strip()

    return utt


def delexicaliseReferenceNumber(sent, metadata):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    if metadata:
        for domain in domains:
            if metadata[domain]['book']['booked']:
                for slot in metadata[domain]['book']['booked'][0]:
                    if slot == 'reference':
                        val = '[' + domain + '_' + slot + ']'
                    else:
                        val = '[' + domain + '_' + slot + ']'
                    key = normalize(metadata[domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with hashtag
                    key = normalize("#" + metadata[domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with ref#
                    key = normalize("ref#" + metadata[domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent


def delexicalse_num(sent):
    # changes to numbers only here
    digitpat = re.compile('\d+')
    sent = re.sub(digitpat, '[value_count]', sent)
    return sent


def e2e_delecalise(utt, dictionary, metadata):
    utt = normalize(utt)
    utt = delexicalise(utt, dictionary)
    utt = delexicaliseReferenceNumber(utt, metadata)
    return delexicalse_num(utt)


if __name__ == '__main__':
    prepareSlotValuesIndependent()
