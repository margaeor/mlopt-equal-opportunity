

function copyToClipboard(text) {
    if (window.clipboardData && window.clipboardData.setData) {
        // Internet Explorer-specific code path to prevent textarea being shown while dialog is visible.
        return window.clipboardData.setData("Text", text);

    }
    else if (document.queryCommandSupported && document.queryCommandSupported("copy")) {
        var textarea = document.createElement("textarea");
        textarea.textContent = text;
        textarea.style.position = "fixed";  // Prevent scrolling to bottom of page in Microsoft Edge.
        document.body.appendChild(textarea);
        textarea.select();
        try {
            return document.execCommand("copy");  // Security exception may be thrown by some browsers.
        }
        catch (ex) {
            console.warn("Copy to clipboard failed.", ex);
            return false;
        }
        finally {
            document.body.removeChild(textarea);
        }
    }
}

function find_colname_from_id(id) {
	var txt = $('.ag-center-cols-container > div[row-id="'+id+'"] > div[col-id="id"]').textContent;
	return txt
}
function copy_values(id) {
	var arr = [];
	var a=$('.ag-full-width-container > div[row-id="detail_'+id+'"] .detail-panel ul'); 
	for ( let i=0;i<a.children.length;i++) arr.push(a.children[i].children[0].textContent);
	var colname = find_colname_from_id(id);
	var s = colname+id+"\tValue\n";
	
	var final_arr = [[colname,'Value']];
	for(let i in arr) final_arr.push(arr[i].split(" -- "));//s += arr[i].replace(" -- ","\t")+"\n";
	//console.log(arr);
	//copyToClipboard(s);
	return final_arr;
}
function get_all_ids() {
	
	var arr = [];
	var a=$('.ag-full-width-container'); 
	for ( let i=0;i<a.children.length;i++) arr.push(a.children[i].attributes['row-id'].textContent.replace("detail_",""));
	return arr;
}

function exportCsv(data) {
	
	
	// Building the CSV from the Data two-dimensional array
	// Each column is separated by ";" and new line "\n" for next row
	var csvContent = '';
	data.forEach(function(infoArray, index) {
	  dataString = infoArray.join('\t');
	  csvContent += index < data.length ? dataString + '\n' : dataString;
	});

	copyToClipboard(csvContent);
}

function export_all() {
	ids =  get_all_ids();
	var results = [];
	var biggest_result = -1;
	for (let i in ids) {
		res = copy_values(ids[i]);
		biggest_result = res.length > biggest_result ? res.length : biggest_result;
		results.push(res);
	}

	for (let i in results) {
		while(results[i].length < biggest_result) {
			results[i].push(['','']);
		}
	}

	var final_result = [];
	for (let i=0; i<biggest_result;i++) {
		let temp = [];
		for(let j in results) {
			temp = temp.concat(results[j][i]);
			if(i!=0)
				temp = temp.concat(['','']);
			else
				temp = temp.concat([results[j][i][0]+'_map',results[j][i][1]+'_map']);
		}
		final_result.push(temp);
	}
	
	exportCsv(final_result);
	return final_result;
}

export_all();
